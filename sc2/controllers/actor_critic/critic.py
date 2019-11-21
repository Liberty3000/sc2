import torch as th

class Critic(th.nn.Module):
    def __init__(self,
                 non_spatial_action_space=549,
                 local_feature_shape=(17,84,84),
                 global_feature_shape=(7,64,64),
                 non_spatial_feature_shape=(1,45)):
        super().__init__()
        self.local_feature_shape=local_feature_shape
        self.global_feature_shape=global_feature_shape
        self.non_spatial_feature_shape=non_spatial_feature_shape
        nonlin = th.nn.ReLU(inplace=True)

        self.local_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d( local_feature_shape[0], 32, 3, stride=2), nonlin,
        th.nn.Conv2d(32, 64, 3, stride=2), nonlin,
        th.nn.Conv2d(64, 96, 3, stride=2), nonlin,
        th.nn.Conv2d(96,128, 3, stride=2), nonlin)
        self.global_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d(global_feature_shape[0], 32, 3, stride=2), nonlin,
        th.nn.Conv2d(32, 64, 3, stride=2), nonlin,
        th.nn.Conv2d(64, 96, 3, stride=2), nonlin,
        th.nn.Conv2d(96,128, 3, stride=2), nonlin)
        self.non_spatial_feature_extractor = th.nn.Sequential(
        th.nn.Linear(non_spatial_feature_shape[-1], 256))

        nfeats = self.nfeatures()

        self.value_head = th.nn.Linear(nfeats, 1)

    def nfeatures(self):
        x = self.local_feature_extractor( th.zeros((1,*self.local_feature_shape)))
        y = self.global_feature_extractor(th.zeros((1,*self.global_feature_shape)))
        z = self.non_spatial_feature_extractor(th.zeros((1,*self.non_spatial_feature_shape)))
        _ = th.cat((x.view(-1),y.view(-1),z.view(-1)),-1).view(-1)
        return _.size()[-1]

    def encode(self, global_features, local_features, non_spatial_features):
        x = self.local_feature_extractor(local_features)
        y = self.global_feature_extractor(global_features)
        z = self.non_spatial_feature_extractor(non_spatial_features)
        bsize = x.size(0)
        _= th.cat((x.view(bsize, -1), y.view(bsize, -1), z.view(bsize, -1)),-1)
        return _

    def __call__(self, global_features, local_features, non_spatial_features, action=None):
        _= self.encode(global_features, local_features, non_spatial_features)

        vhead = self.value_head(_)
        return vhead
