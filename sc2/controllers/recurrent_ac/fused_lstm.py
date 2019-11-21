import numpy as np, torch as th

class FusedLSTM(th.nn.Module):
    def __init__(self,
                 non_spatial_actions=549,
                 local_feature_shape=(17,84,84),
                 global_feature_shape=(7,64,64),
                 non_spatial_feature_shape=(1,45)):
        super().__init__()
        self.non_spatial_actions=non_spatial_actions
        self.local_feature_shape=local_feature_shape
        self.global_feature_shape=global_feature_shape
        self.non_spatial_feature_shape=non_spatial_feature_shape
        self.nonlin = th.nn.ELU()
        hidden = 256

        shared_construct = lambda inch:[
        th.nn.Conv2d(inch, 24, 3, stride=2, padding=1), self.nonlin,
        th.nn.Conv2d(  24, 48, 3, stride=2, padding=1), self.nonlin,
        th.nn.Conv2d(  48, 96, 3, stride=2, padding=1), self.nonlin,
        th.nn.Conv2d(  96, 32, 3, stride=2, padding=1), th.nn.MaxPool2d((3,3))]

        # input streams
        self.minimap_stream = th.nn.Sequential(*shared_construct( 7).copy())
        self.screen_stream  = th.nn.Sequential(*shared_construct(17).copy())
        self.nonspt_stream  = th.nn.Sequential(
        th.nn.Linear(non_spatial_feature_shape[-1], 128))
        # fused recurrent layer
        self.recurrent_layer = th.nn.LSTMCell(self.feature_map_size(), hidden)
        # value function
        self.global_critic = th.nn.Linear(hidden, 1)
        # action-value function
        self.global_non_spatial_actor  = th.nn.Linear(hidden, non_spatial_actions)
        # spatial action-value function
        self.global_spatial_1x1 = th.nn.Conv2d(7, 1, 1)
        self.global_spatial_3x3 = th.nn.Conv2d(7, 1, 3, padding=1)
        self.global_spatial_actor = th.nn.Linear(hidden, 4096)

    def init_recurrence(self):
        return (th.zeros(1,256),th.zeros(1,256))

    def feature_map_size(self):
        return 32 * 3 * 3

    def __call__(self, x, y, z, recurrence):
        (hx, cx) = recurrence
        bsize = x.size(0)

        a = self.minimap_stream(x).view(bsize, -1)
        b = self.screen_stream(y).view(bsize, -1)
        c = self.nonspt_stream(z).view(bsize, -1)

        _= self.nonlin(th.cat((a,b,c),dim=-1))

        recurrence,_ = self.recurrent_layer(_, recurrence)

        x1 = self.global_spatial_1x1(x)
        x2 = self.global_spatial_3x3(x)
        spatial_logits = th.matmul(x1, x2).view(bsize, -1)

        state_value = self.global_critic(hx)
        non_spatial_logits = self.global_non_spatial_actor(hx)
        spatial_logits = self.global_spatial_actor(hx)

        return state_value, non_spatial_logits, spatial_logits
