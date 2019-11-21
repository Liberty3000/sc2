import numpy as np, torch as th

class FullyConv(th.nn.Module):
    def __init__(self,
                 non_spatial_actions=549,
                 local_feature_shape=(17,84,84),
                 global_feature_shape=(7,64,64),
                 non_spatial_feature_shape=(1,45),
                 nonlin=th.nn.ReLU(True),
                 dropout=0.):
        super().__init__()
        self.non_spatial_actions=non_spatial_actions
        self.local_feature_shape=local_feature_shape
        self.global_feature_shape=global_feature_shape
        self.non_spatial_feature_shape=non_spatial_feature_shape

        self.local_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d( local_feature_shape[0], 16, 3, stride=2), nonlin,
        th.nn.Conv2d(16, 32, 5, stride=3, padding=2), nonlin)

        self.global_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d(global_feature_shape[0], 16, 5, stride=2), nonlin,
        th.nn.Conv2d(16, 32, 3, stride=2), nonlin)

        self.non_spatial_feature_extractor = th.nn.Sequential(
        th.nn.Linear(non_spatial_feature_shape[-1], 196))

        nfeats = self.nfeatures()

        self.value_head = th.nn.Sequential(
        th.nn.Dropout(dropout),
        th.nn.Conv2d(nfeats, 1, 14))

        self.non_spatial_policy_head = th.nn.Sequential(
        th.nn.Dropout(dropout),
        th.nn.Conv2d(nfeats, non_spatial_actions, 14),
        th.nn.Softmax(dim=1))

        self.xy_head = th.nn.Sequential(
        th.nn.Dropout(dropout),
        th.nn.Conv2d(global_feature_shape[0], 1, 3, 1, 1))

    def nfeatures(self):
        local_features = th.zeros((1, *self.local_feature_shape))
        global_features = th.zeros((1, *self.global_feature_shape))
        non_spatial_features = th.zeros((1, *self.non_spatial_feature_shape))
        x = self.global_feature_extractor(global_features)
        y = self.local_feature_extractor(local_features)
        z = self.non_spatial_feature_extractor(non_spatial_features)
        _= th.cat((x, y, z.view(-1,1,14,14)), dim=1)
        return _.size(1)

    def __call__(self, global_features, local_features, non_spatial_features):
        x = self.global_feature_extractor(global_features)
        y = self.local_feature_extractor(local_features)
        z = self.non_spatial_feature_extractor(non_spatial_features)
        _= th.cat((x, y, z.view(-1,1,14,14)), dim=1)

        state_value = self.value_head(_)

        spatial_logits = self.xy_head(global_features)

        non_spatial_logits = self.non_spatial_policy_head(_).squeeze()

        return state_value, non_spatial_logits, spatial_logits
