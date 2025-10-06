from models.tcn import TCN
from .base_module import BaseModule

class TCNModule(BaseModule):
    def __init__(
        self,
        num_features,
        num_channels,
        kernel_size,
        dropout,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        window_size=14,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss='rmse',
        **kwargs
    ):
        super().__init__(
            lr=lr,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            weight_decay=weight_decay,
            logger_type=logger_type,
            y_key=y_key,
            optimizer=optimizer,
            mode=mode,
            window_size=window_size,
            loss=loss,
        )
        
        self.model = TCN(
            num_features=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            **kwargs
        )