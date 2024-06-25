
import torch
import torch.nn.functional as F
import pytorch_lightning as pl



DEFAULT_FRICTION_PENALTY = 1.0
DEFAULT_ACCELERATION_PENALTY = 2.0
DEFAULT_VELOCITY_PENALTY = 2.0
DEFAULT_ENERGY_PENALTY = 2.0
DEFAULT_KL_BETA = 0.01
DEFAULT_RANDOM_FACTOR = 0.1


# Dynamical Lightning module
class DynamicalPredictor(pl.LightningModule):
    def __init__(self, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY):
        super().__init__()
        self.friction_penalty = friction_penalty
        self.acceleration_penalty = acceleration_penalty
        self.velocity_penalty = velocity_penalty
        self.energy_penalty = energy_penalty

        self.save_hyperparameters()

    def friction_force(self, v):
        return -self.friction_penalty * v
    
    def acceleration_loss(self, a):
        return torch.mean(a ** 2)
    
    def velocity_loss(self, v, a):
        return torch.mean((v + a) ** 2)

    def _smooth_abs(self, x, beta=0.1):
        return F.smooth_l1_loss(x, torch.zeros_like(x), beta, reduce=False, reduction='none')
    
    def energy_loss(self, x, v, a_target, a_pred):
        # Compute the work done by the true application of the force (true past values)
        future_x = x[:, 1:, :]
        past_x = x[:, :-1, :]
        work = self._smooth_abs(torch.sum(a_target[:,:-1,:] * (future_x - past_x), dim=2)) # Work done by the application of the force (batch_size, lookback-1)

        for i in range(1, work.shape[1]):
            work[:,i] += work[:,i-1]
        work = torch.cat([torch.zeros(work.shape[0], 1, device=work.device), work], dim=1) # Cumulative work done by the application of the force (batch_size, lookback)

        # Compute the energy spent by the system (predicted future values given true past work)
        x_pred = x + v + 0.5 * a_pred
        energy_pred = work + self._smooth_abs(torch.sum(a_pred * (x_pred - x), dim=2)) # Cumulative predicted energy spent by the system (batch_size, lookback)
        
        scaling_factor = torch.arange(1, x.shape[1]+1, device=x.device).float().reshape((1, -1)) # Linear scaling factor reducing the effect of early predictions with low work (batch_size, lookback)
        return torch.mean(scaling_factor * energy_pred**2)
    
    def compute_losses(self, y_pred, y, x, v, log_step = None):
        prediction_loss = torch.nn.functional.mse_loss(y_pred, y)
        acceleration_loss = self.acceleration_loss(y_pred)
        velocity_loss = self.velocity_loss(v, y_pred)
        energy_loss = self.energy_loss(x, v, y, y_pred)

        loss = prediction_loss + self.acceleration_penalty * acceleration_loss + self.velocity_penalty * velocity_loss + self.energy_penalty * energy_loss

        if log_step:
            self.log(f'{log_step}_loss', loss)
            self.log(f'{log_step}_prediction_loss', prediction_loss.detach())
            self.log(f'{log_step}_acceleration_loss', acceleration_loss.detach())
            self.log(f'{log_step}_velocity_loss', velocity_loss.detach())
            self.log(f'{log_step}_energy_loss', energy_loss.detach())
            self.log(f'{log_step}_predicted_force', torch.mean(y_pred).detach())
            self.log(f'{log_step}_predicted_force_std', torch.std(y_pred).detach())
        
        return loss

    
    def training_step(self, batch, batch_idx):
        x, v, y, i = batch
        y_pred = self(x, velocity=v)
        friction = self.friction_force(v)
        y_pred = y_pred + friction

        loss = self.compute_losses(y_pred, y, x, v, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, v, y, i = batch
        y_pred = self(x, velocity=v)
        friction = self.friction_force(v)
        y_pred = y_pred + friction
        
        loss = self.compute_losses(y_pred, y, x, v, 'val')
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, v, y, i = batch
        y_pred = self(x, velocity=v)
        y_pred = y_pred + self.friction_force(v)
    
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)



class DynLSTMPredictor(DynamicalPredictor):
    def __init__(self, lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_size=256, num_layers=3, include_velocity=True):
        super().__init__(friction_penalty, acceleration_penalty, velocity_penalty, energy_penalty)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dimensions = 3 # x, y, z
        self.include_velocity = include_velocity # Additional velocity input with same dimensions

        # self.input_batch_norm = torch.nn.BatchNorm1d(self.dimensions + (self.dimensions if include_velocity else 0))
        self.lstm = torch.nn.LSTM(input_size=self.dimensions * (2 if self.include_velocity else 1), hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.output_batch_norm = torch.nn.BatchNorm1d(self.hidden_size)
        self.linear = torch.nn.Linear(hidden_size, self.dimensions) # Output is the force applied on the system at the next timestep with same dimensions
        self.save_hyperparameters()

    def forward(self, x, return_latent=False, velocity=None, **kwargs):
        if self.include_velocity and velocity is not None:
            x = torch.cat([x, velocity], dim=2)

        x_shape = x.shape # [batch_size, lookback, dimensions]
        
        # x = self.input_batch_norm(x.reshape((-1, self.dimensions)))
        # x = x.reshape(x_shape)

        latents, _ = self.lstm(x)
        # latents = self.output_batch_norm(latents.reshape((-1, self.hidden_size)))
        x = self.linear(latents)
        # x = x.reshape(x_shape)

        if return_latent:
            return x, latents
        return x
    

class DynMLPPredictor(DynamicalPredictor):
    def __init__(self, lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_size=128, num_layers=2, include_velocity=True):
        super().__init__(friction_penalty, acceleration_penalty, velocity_penalty, energy_penalty)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dimensions = 3 # x, y, z
        self.include_velocity = include_velocity # Additional velocity input with same dimensions

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.dimensions * self.lookback * (2 if self.include_velocity else 1), self.hidden_size),
            torch.nn.ReLU()
        )

        hidden_layers = []
        for _ in range(num_layers-1):
            hidden_layers.append(torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU()
            ))
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)

        self.output_layer = torch.nn.Linear(self.hidden_size, self.dimensions) # Output is the force applied on the system at the next timestep with same dimensions

        self.save_hyperparameters()

    def forward(self, x, return_latent=False, velocity=None, **kwargs):
        x_shape = x.shape # [batch_size, lookback, dimensions]

        if self.include_velocity and velocity is not None:
            x = torch.cat([x, velocity], dim=2)
        
        lookback = min(self.lookback, x_shape[1])
        latents = torch.zeros(x_shape[0], lookback, self.hidden_size, device=x.device)
        for i in range(1,lookback+1):
            x_i = x[:,:i,:]
            lookback_i = min(self.lookback, x_i.shape[1])

            if lookback_i < self.lookback:
                x_i = torch.cat([torch.zeros(x_shape[0], self.lookback - lookback_i, self.dimensions * (2 if self.include_velocity else 1), device=x_i.device), x_i], dim=1)

            latents[:,i-1,:] = self.input_layer(x_i.reshape((-1, self.dimensions * self.lookback* (2 if self.include_velocity else 1))))

        for layer in self.hidden_layers: # Residual connections
            latents = latents + layer(latents)

        x = self.output_layer(latents)
        x = x.reshape(x_shape)

        if return_latent:
            return x, latents
        return x
    

class DynTransformerPredictor(DynamicalPredictor):
    def __init__(self, lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_size=192, nhead=3, num_encoder_layers=2, num_decoder_layers=2, include_velocity=True):
        super().__init__(friction_penalty, acceleration_penalty, velocity_penalty, energy_penalty)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dimensions = 3 # x, y, z
        self.include_velocity = include_velocity # Additional velocity input with same dimensions

        # self.input_batch_norm = torch.nn.BatchNorm1d(self.dimensions)
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.dimensions * (2 if self.include_velocity else 1), self.hidden_size),
            torch.nn.ReLU()
        )
        self.transformer = torch.nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        # self.output_batch_norm = torch.nn.BatchNorm1d(self.hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, self.dimensions) # Output is the force applied on the system at the next timestep with same dimensions
        self.save_hyperparameters()
    
    def forward(self, x, return_latent=False, velocity=None, **kwargs):
        if self.include_velocity and velocity is not None:
            x = torch.cat([x, velocity], dim=2)

        x_shape = x.shape # [batch_size, lookback, dimensions]
        lookback = min(self.lookback, x_shape[1])
        
        # x = self.input_batch_norm(x.reshape((-1, self.dimensions)))
        # x = x.reshape(x_shape)
        x = self.input_layer(x)
        
        latents = torch.zeros_like(x)
        for i in range(1,lookback+1):
            output_i = self.transformer(x[:,:i,:], x[:,:i,:])
            latents[:,i-1,:] = output_i[:,-1,:]
        
        # outputs = self.output_batch_norm(outputs.reshape((-1, self.hidden_size)))
        outputs = self.output_layer(latents)
        # outputs = outputs.reshape(x_shape)

        if return_latent:
            return outputs, latents
        return outputs
    



class DynVariationalPredictor(DynamicalPredictor):
    def __init__(self, inner, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, beta=DEFAULT_KL_BETA, random_factor=DEFAULT_RANDOM_FACTOR):
        super().__init__(friction_penalty, acceleration_penalty, velocity_penalty, energy_penalty)
        self.inner = inner
        self.log_var_fn = torch.nn.Linear(self.inner.hidden_size, self.inner.dimensions)
        self.beta = beta
        self.random_factor = random_factor

        self.save_hyperparameters()

    def forward(self, x, return_latent=False, velocity=None, **kwargs):
        mu, latents = self.inner.forward(x, return_latent=True, velocity=velocity)
        log_var = self.log_var_fn(latents)

        sample = self.random_factor * torch.randn_like(mu)
        sample = mu + sample * torch.exp(0.5 * log_var)

        if return_latent:
            return sample, mu, log_var, latents
        return sample
    
    def kl_loss(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))
    
    def training_step(self, batch, batch_idx):
        x, v, y, i = batch
        y_pred, mu, log_var, _ = self(x, return_latent=True, velocity=v)
        friction = self.friction_force(v)
        y_pred = y_pred + friction

        prediction_loss = torch.nn.functional.mse_loss(y_pred, y)
        kl_loss = self.kl_loss(mu, log_var)
        acceleration_loss = self.acceleration_loss(y_pred)
        velocity_loss = self.velocity_loss(v, y_pred)
        energy_loss = self.energy_loss(x, v, y, y_pred)
        loss = prediction_loss + self.acceleration_penalty * acceleration_loss + self.velocity_penalty * velocity_loss + self.energy_penalty * energy_loss + self.beta * kl_loss
        self.log('train_loss', loss)
        self.log('train_prediction_loss', prediction_loss.detach())
        self.log('train_acceleration_loss', acceleration_loss.detach())
        self.log('train_velocity_loss', velocity_loss.detach())
        self.log('train_energy_loss', energy_loss.detach())
        self.log('train_kl_loss', kl_loss.detach())
        self.log('train_friction_force', torch.mean(friction).detach())
        self.log('train_friction_force_std', torch.std(friction).detach())
        self.log('train_predicted_force', torch.mean(y_pred).detach())
        self.log('train_predicted_force_std', torch.std(y_pred).detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, v, y, i = batch
        y_pred, mu, log_var, _ = self(x, return_latent=True, velocity=v)
        friction = self.friction_force(v)
        y_pred = y_pred + friction
        
        prediction_loss = torch.nn.functional.mse_loss(y_pred, y)
        kl_loss = self.kl_loss(mu, log_var)
        acceleration_loss = self.acceleration_loss(y_pred)
        velocity_loss = self.velocity_loss(v, y_pred)
        energy_loss = self.energy_loss(x, v, y, y_pred)
        loss = prediction_loss + self.acceleration_penalty * acceleration_loss + self.velocity_penalty * velocity_loss + self.energy_penalty * energy_loss + self.beta * kl_loss
        self.log('val_loss', loss)
        self.log('val_prediction_loss', prediction_loss.detach())
        self.log('val_acceleration_loss', acceleration_loss.detach())
        self.log('val_velocity_loss', velocity_loss.detach())
        self.log('val_energy_loss', energy_loss.detach())
        self.log('val_kl_loss', kl_loss.detach())
        self.log('val_friction_force', torch.mean(friction).detach())
        self.log('val_friction_force_std', torch.std(friction).detach())
        self.log('val_predicted_force', torch.mean(y_pred).detach())
        self.log('val_predicted_force_std', torch.std(y_pred).detach())
        return loss
    

def DynVariationalMLPPredictor(lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_size=128, num_layers=2, beta=DEFAULT_KL_BETA):
    return DynVariationalPredictor(DynMLPPredictor(lookback, hidden_size=hidden_size, num_layers=num_layers, include_velocity=True), beta=beta, friction_penalty=friction_penalty, acceleration_penalty=acceleration_penalty, velocity_penalty=velocity_penalty, energy_penalty=energy_penalty)

def DynVariationalLSTMPredictor(lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_size=256, num_layers=3, beta=DEFAULT_KL_BETA):
    return DynVariationalPredictor(DynLSTMPredictor(lookback, hidden_size=hidden_size, num_layers=num_layers, include_velocity=True), beta=beta, friction_penalty=friction_penalty, acceleration_penalty=acceleration_penalty, velocity_penalty=velocity_penalty, energy_penalty=energy_penalty)

def DynVariationalTransformerPredictor(lookback, friction_penalty=DEFAULT_FRICTION_PENALTY, acceleration_penalty=DEFAULT_ACCELERATION_PENALTY, velocity_penalty=DEFAULT_VELOCITY_PENALTY, energy_penalty=DEFAULT_ENERGY_PENALTY, hidden_size=192, nhead=3, num_encoder_layers=2, num_decoder_layers=2, beta=DEFAULT_KL_BETA):
    return DynVariationalPredictor(DynTransformerPredictor(lookback, hidden_size=hidden_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, include_velocity=True), beta=beta, friction_penalty=friction_penalty, acceleration_penalty=acceleration_penalty, velocity_penalty=velocity_penalty, energy_penalty=energy_penalty)



DYNAMIC_MODELS = {
    "dynamical_lstm": DynLSTMPredictor,
    "dynamical_mlp": DynMLPPredictor,
    "dynamical_transformer": DynTransformerPredictor,
    "dynamical_variational": DynVariationalPredictor,
    "dynamical_variational_lstm": DynVariationalLSTMPredictor,
    "dynamical_variational_mlp": DynVariationalMLPPredictor,
    "dynamical_variational_transformer": DynVariationalTransformerPredictor
}