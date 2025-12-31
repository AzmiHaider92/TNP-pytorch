''' Implementation of FlowNP model (FNP).'''

import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from attrdictionary import AttrDict
from models.modules import build_mlp
from flow_matching.solver import ODESolver


def comp_posenc(dim_posenc, pos):
    shp = pos.shape
    omega = torch.arange(dim_posenc // 2, dtype=torch.float).to(pos.device)
    omega = torch.pi * 2 ** (omega - 2)
    out = pos[:, :, :, None] * omega[None, None, None, :]
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    emb = torch.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb.reshape(shp[0], shp[1], shp[-1] * dim_posenc)


class FNP(nn.Module):
    def __init__(
            self,
            dim_x,
            dim_y,
            dim_posenc,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            timesteps=128,
            drop_y=0.5
    ):
        super(FNP, self).__init__()

        self.timesteps = timesteps
        self.drop_y = drop_y
        self.dim_posenc = dim_posenc

        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y)
        )

        self.embedder = build_mlp((dim_x + 1) * dim_posenc + dim_y, d_model, d_model, emb_depth)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        def drop(y):
            y_dropped = torch.randn_like(y)
            not_drop_ids = torch.rand_like(y) > self.drop_y
            y_dropped[not_drop_ids] = y[not_drop_ids]
            return y_dropped

        def encode(xc, yc, xt, yt):
            # follow tnp convention of dropping rewards for context
            #x_y_ctx = torch.cat((xc, yc), dim=-1)
            yc_dropped = drop(yc)
            x_y_ctx = torch.cat((xc, yc_dropped), dim=-1)

            x_y_tar = torch.cat((xt, yt), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar), dim=1)

            num_ctx, num_tar = xc.shape[1], xt.shape[1]
            num_all = num_ctx + num_tar
            mask = torch.zeros(num_all, num_all, device='cuda')

            embeddings = self.embedder(inp)
            encoded = self.encoder(embeddings, mask=mask)[:, -num_tar:]
            out = self.predictor(encoded)
            return out

        self.encode = encode

        class Predict_Velocity(nn.Module):
            def forward(self, x: torch.Tensor, t: torch.Tensor, batch=None):
                yt = x.reshape(batch.xt.shape[0], batch.xt.shape[1], -1)
                if t.dim() == 0:
                    t = t.repeat((yt.shape[0], yt.shape[1], 1)).to(x.device)
                xc = torch.cat((batch.xc, torch.ones(
                    list(batch.xc.shape[:-1]) + [1]).to(x.device)), dim=-1)
                xt = torch.cat((batch.xt, t), dim=-1)
                pred = encode(xc=comp_posenc(dim_posenc, xc),
                              xt=comp_posenc(dim_posenc, xt),
                              yc=batch.yc, yt=yt)
                p = pred.reshape(x.shape)
                return p

        self.predict_velocity = Predict_Velocity()
        self.solver = ODESolver(velocity_model=self.predict_velocity)

    def forward(self, batch, reduce_ll=True):
        y0 = torch.randn_like(batch.yt).to(batch.yt.device)
        t = torch.rand(size=(y0.shape[0], y0.shape[1], 1)).to(y0.device)
        yt = t * batch.yt + (1 - t) * y0
        pred = self.predict_velocity(yt, t, batch)

        outs = AttrDict()
        outs.loss = nn.MSELoss()(pred, batch.yt - y0)
        return outs

    def predict(self, xc, yc, xt, num_samples=30, return_samples=False):

        batch_size = xc.shape[0]
        num_target = xt.shape[1]
        T = self.timesteps
        xc = xc.repeat((num_samples, 1, 1))
        yc = yc.repeat((num_samples, 1, 1))
        xt = torch.cat((xc, xt.repeat((num_samples, 1, 1))), dim=1)
        yt = torch.randn((num_samples * batch_size, xt.shape[1], yc.shape[2])).to(xt.device)
        xct = torch.cat((xc, torch.ones((xc.shape[0], xc.shape[1], 1)).to(xc.device)), dim=-1)
        for t in range(T):
            tt = torch.tensor(t / T).repeat((yt.shape[0], yt.shape[1], 1)).to(yt.device)
            xtt = torch.cat((xt, tt), dim=-1)
            pred = self.encode(xc=comp_posenc(self.dim_posenc, xct),
                               xt=comp_posenc(self.dim_posenc, xtt),
                               yc=yc, yt=yt)
            alpha = 1 + t / T * (1 - t / T)
            sigma = 0.2 * (t / T * (1 - t / T)) ** 0.5
            yt += (alpha * pred + sigma * torch.randn_like(yt).to(yt.device)) / T

        samples = yt[:, xc.shape[1]:].reshape(num_samples, batch_size, num_target, yc.shape[2])
        # empirical mean/std across samples -> [B, Nt, Dy]
        loc = samples.mean(dim=0, keepdim=True)
        scale = samples.std(dim=0, unbiased=False, keepdim=True)

        outs = AttrDict()
        outs.loc = loc
        outs.scale = scale
        outs.ys = Normal(outs.loc, outs.scale)
        if return_samples:
            outs.samples = samples  # [S, B, Nt, Dy]

        return outs
