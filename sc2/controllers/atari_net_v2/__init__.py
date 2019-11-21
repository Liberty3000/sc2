import torch as th

class AtariNet(th.nn.Module):
    def __init__(self,
                 non_spatial_action_space=549,
                 local_feature_shape=(17,84,84),
                 global_feature_shape=(7,64,64),
                 non_spatial_feature_shape=(1,45),
                 dropout=0.1):

        super().__init__()
        self.local_feature_shape=local_feature_shape
        self.global_feature_shape=global_feature_shape
        self.non_spatial_feature_shape=non_spatial_feature_shape
        nonlin = th.nn.ReLU(inplace=True)

        self.local_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d( local_feature_shape[0], 16, 8, stride=4), nonlin,
        th.nn.Conv2d(16, 32, 4, stride=2), nonlin)

        self.global_feature_extractor = th.nn.Sequential(
        th.nn.Conv2d(global_feature_shape[0], 16, 8, stride=4), nonlin,
        th.nn.Conv2d(16, 32, 4, stride=2), nonlin)

        self.non_spatial_feature_extractor = th.nn.Sequential(
        th.nn.Linear(non_spatial_feature_shape[-1], 256))

        nfeats = self.nfeatures()

        self.value_head = th.nn.Linear(nfeats, 1)

        self.x_head = th.nn.Sequential(
        th.nn.Dropout(dropout),
        th.nn.Linear(nfeats, global_feature_shape[-2]),
        th.nn.Softmax(dim=1))

        self.y_head = th.nn.Sequential(
        th.nn.Dropout(dropout),
        th.nn.Linear(nfeats, global_feature_shape[-1]),
        th.nn.Softmax(dim=1))

        self.non_spatial_policy_head = th.nn.Sequential(
        th.nn.Linear(nfeats,non_spatial_action_space),
        th.nn.Softmax(dim=1))

    def nfeatures(self):
        x = self.local_feature_extractor( th.zeros((1,*self.local_feature_shape)))
        y = self.global_feature_extractor(th.zeros((1,*self.global_feature_shape)))
        z = self.non_spatial_feature_extractor(th.zeros((1,*self.non_spatial_feature_shape)))
        _ = th.cat((x.view(-1),y.view(-1),z.view(-1)),-1).view(-1)
        return _.size()[-1]

    def encode(self, global_features, local_features, non_spatial_features):
        x = self.local_feature_extractor(local_features)
        y = self.global_feature_extractor(global_features)
        #print(non_spatial_features.size())
        z = self.non_spatial_feature_extractor(non_spatial_features)
        bsize = x.size(0)
        # state representation
        _= th.cat((x.view(bsize, -1), y.view(bsize, -1), z.view(bsize, -1)),-1)
        return _

    def __call__(self, global_features, local_features, non_spatial_features):
        _= self.encode(global_features, local_features, non_spatial_features)

        # these could be paralellized
        vhead = self.value_head(_)
        xhead = th.distributions.Categorical(self.x_head(_)).sample()
        yhead = th.distributions.Categorical(self.y_head(_)).sample()
        zhead = self.non_spatial_policy_head(_)

        return vhead, zhead, (xhead, yhead)
