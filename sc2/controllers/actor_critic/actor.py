import torch as th

class Actor(th.nn.Module):
    def __init__(self,
                 non_spatial_action_space=549,
                 local_feature_shape=(17,84,84),
                 global_feature_shape=(7,64,64),
                 non_spatial_feature_shape=(1,45)):
        super().__init__()
        self.local_feature_shape=local_feature_shape
        self.global_feature_shape=global_feature_shape
        self.non_spatial_feature_shape=non_spatial_feature_shape
        self.nonlin = th.nn.ReLU(inplace=True)

        self.local_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d( local_feature_shape[0], 32, 3, stride=2), self.nonlin,
        th.nn.Conv2d(32, 64, 3, stride=2), self.nonlin,
        th.nn.Conv2d(64, 96, 3, stride=2), self.nonlin,
        th.nn.Conv2d(96,128, 3, stride=2))
        self.global_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d(global_feature_shape[0], 32, 3, stride=2), self.nonlin,
        th.nn.Conv2d(32, 64, 3, stride=2), self.nonlin,
        th.nn.Conv2d(64, 96, 3, stride=2), self.nonlin,
        th.nn.Conv2d(96,128, 3, stride=2))
        self.non_spatial_feature_extractor = th.nn.Sequential(
        th.nn.Linear(non_spatial_feature_shape[-1], 256))

        nfeats = self.nfeatures()

        self.spatial_policy_x = th.nn.Sequential(
        th.nn.Linear(nfeats, global_feature_shape[-2]),
        th.nn.Softmax(dim=1))

        self.spatial_policy_y = th.nn.Sequential(
        th.nn.Linear(nfeats, global_feature_shape[-1]),
        th.nn.Softmax(dim=1))

        self.non_spatial_policy= th.nn.Sequential(
        th.nn.Linear(nfeats,non_spatial_action_space),
        th.nn.Softmax(dim=1))

    def nfeatures(self):
        x = self.global_feature_extractor(th.zeros((1,*self.global_feature_shape)))
        y = self.local_feature_extractor( th.zeros((1,*self.local_feature_shape)))
        z = self.non_spatial_feature_extractor(th.zeros((1,*self.non_spatial_feature_shape)))
        _ = th.cat((x.view(-1),y.view(-1),z.view(-1)),-1).view(-1)
        return _.size()[-1]

    def encode(self, global_features, local_features, non_spatial_features):
        x = self.global_feature_extractor(global_features)
        y = self.local_feature_extractor(local_features)
        z = self.non_spatial_feature_extractor(non_spatial_features)
        bsize = x.size(0)
        _= th.cat((x.view(bsize, -1), y.view(bsize, -1), z.view(bsize, -1)),-1)
        return _

    def __call__(self, global_features, local_features, non_spatial_features):
        _= self.encode(global_features, local_features, non_spatial_features)

        _= self.nonlin(_)

        xlogits = self.spatial_policy_x(_)
        ylogits = self.spatial_policy_y(_)

        non_spatial_logits = self.non_spatial_policy(_).squeeze()

        return non_spatial_logits, (xlogits, ylogits)
