import torch as th

class AtariNet(th.nn.Module):
    def __init__(self,
                 non_spatial_actions=549,
                 local_feature_shape=(17,84,84),
                 global_feature_shape=(7,64,64),
                 non_spatial_feature_shape=(1,45),
                 nonlin=th.nn.ReLU(True),
                 dropout=0.10):

        super().__init__()
        self.nonlin = nonlin
        self.local_feature_shape=local_feature_shape
        self.global_feature_shape=global_feature_shape
        self.non_spatial_feature_shape=non_spatial_feature_shape

        self.local_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d( local_feature_shape[0], 16, 3, stride=2),
        self.nonlin,
        th.nn.Conv2d(16, 32, 5, stride=3, padding=2))

        self.global_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d(global_feature_shape[0], 16, 5, stride=2),
        self.nonlin,
        th.nn.Conv2d(16, 32, 3, stride=2), nonlin)

        self.non_spatial_feature_extractor = th.nn.Sequential(
        th.nn.Linear(non_spatial_feature_shape[-1], 256),
        th.nn.Tanh())

        nfeats = self.nfeatures()

        self.value_estimate = th.nn.Linear(nfeats, 1)

        self.spatial_policy_x = th.nn.Sequential(
        th.nn.Linear(nfeats, global_feature_shape[-2]),
        th.nn.Softmax(dim=1))

        self.spatial_policy_y = th.nn.Sequential(
        th.nn.Linear(nfeats, global_feature_shape[-1]),
        th.nn.Softmax(dim=1))

        self.non_spatial_policy= th.nn.Sequential(
        th.nn.Linear(nfeats, non_spatial_actions),
        th.nn.Softmax(dim=1))

    def nfeatures(self):
        x = self.global_feature_extractor(th.zeros((1, *self.global_feature_shape)))
        y = self.local_feature_extractor(th.zeros((1, *self.local_feature_shape)))
        z = self.non_spatial_feature_extractor(th.zeros((1, *self.non_spatial_feature_shape)))
        _= th.cat((x.view(1,-1), y.view(1,-1), z.view(1,-1)), dim=1)
        return _.size(1)

    def __call__(self, global_features, local_features, non_spatial_features):
        bsize = global_features.size(0)
        x = self.global_feature_extractor(global_features)
        y = self.local_feature_extractor(local_features)
        z = self.non_spatial_feature_extractor(non_spatial_features)
        _= th.cat((x.view(bsize,-1), y.view(bsize,-1), z.view(bsize,-1)), dim=1)

        _= self.nonlin(_)

        state_value = self.value_estimate(_)

        xlogits = self.spatial_policy_x(_)
        ylogits = self.spatial_policy_y(_)

        non_spatial_logits = self.non_spatial_policy(_).squeeze()

        return state_value, non_spatial_logits, (xlogits, ylogits)
