"""
Main class carrying various network modules (e.g., for different modalities)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from place_from_pick_learning.networks import *
from place_from_pick_learning.utils.learning_utils import RGB_IMG_KEYS, FLOAT_IMG_KEYS, IMG_KEYS


def build_fc_modules(
    input_size,
    linear_layer_list,
    activation_func_list,
    output_size,
    output_activation=None
):
    '''
    Return an instance of nn.Sequential
    '''
    fc_module = None
    activation_func_class_list = None
    if activation_func_list is not None:
        assert len(linear_layer_list) == len(activation_func_list)
        activation_func_class_list = [getattr(nn, act_func) for act_func in activation_func_list]

    if len(linear_layer_list) > 0:

        fc_module = [nn.Linear(input_size, linear_layer_list[0],)]
        if activation_func_class_list:
            fc_module.append(activation_func_class_list[0]())

        for i in range(1, len(linear_layer_list)):

            fc_module.append(
                nn.Linear(
                    linear_layer_list[i-1],
                    linear_layer_list[i]
                )
            )

            if activation_func_class_list:
                fc_module.append(activation_func_class_list[i]())


    #add the final layer
    if len(linear_layer_list) > 0:
        fc_module.append(nn.Linear(linear_layer_list[-1], output_size))
    else:
        fc_module = [nn.Linear(input_size, output_size)]

    if output_activation is not None:
        fc_module.append(getattr(nn, output_activation)())

    fc_module = nn.Sequential(*fc_module)

    return fc_module

class MultimodalBC(nn.Module):
    '''
    A monstrosity of various model options to experiment conveniently.
    Sensor:
        'ee' : 7x1
        'q' : 7x1
        'gripper' : 2x1
        'rgb' :
        'depth' :
        'tactile' :

    '''
    def __init__(
        self,
        rotation_representation,
        shared_net_fc_layer_list,
        shared_net_activation_func_list,
        shared_net_embedding_dim,
        action_distribution,
        n_kps,
        n_frame_stacks,
        use_shift_augmentation,
        use_depth,
        use_spatial_encoding,
        use_gripper,
        sequential_model,
        gripper_fc_layer_list,
        gripper_activation_func_list,
        gripper_embedding_dims,
        use_ee,
        ee_fc_layer_list,
        ee_activation_func_list,
        ee_embedding_dims,
        freeze_resnet,
        use_input_coord_conv,
        use_low_noise_eval
    ):
        super(MultimodalBC, self).__init__()
        self.use_shift_augmentation = use_shift_augmentation
        self.use_depth = use_depth
        self.use_spatial_encoding = use_spatial_encoding
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.sequential_model = sequential_model
        self.rotation_representation = rotation_representation
        self.n_frame_stacks = n_frame_stacks
        self.n_kps = n_kps

        if self.rotation_representation in ["quat", "axisa"]:
            rot_dim = 4
        elif self.rotation_representation in ["rvec", "euler"]:
            rot_dim = 3

        # Visual
        self.rgb_encoder = ResNet18Conv(
            pretrained=True,
            input_channel=3 * (n_frame_stacks + 1),
            input_coord_conv=use_input_coord_conv
        )
        if freeze_resnet:
            for param in self.rgb_encoder.parameters():
                param.requires_grad = False

        # Spatial encoder
        # Reduce channels to amount of keypoints
        if use_spatial_encoding:
            self.kp_filter = torch.nn.Conv2d(512, n_kps, kernel_size=1)
            self.spatial_softmax = SpatialSoftArgmax()
            fc_inp_size = 2 * n_kps
        else:
            self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
            fc_inp_size = 512

        if use_depth:
            self.depth_encoder = ResNet18Conv(
                pretrained=True,
                input_channel=1 * (n_frame_stacks + 1)
            )
            if self.use_spatial_encoding:
                fc_inp_size += 2 * n_kps
            else:
                fc_inp_size += 512

        # Extra modalities
        if use_ee:
            self.ee_net = build_fc_modules(
                (3 + rot_dim) * (n_frame_stacks + 1),
                ee_fc_layer_list,
                ee_activation_func_list,
                ee_embedding_dims
            )
            fc_inp_size += ee_embedding_dims

        if use_gripper:
            self.gripper_net = build_fc_modules(
                2 * (n_frame_stacks + 1),
                gripper_fc_layer_list,
                gripper_activation_func_list,
                gripper_embedding_dims
            )
            fc_inp_size += gripper_embedding_dims

        self.shared_net = build_fc_modules(
            fc_inp_size,
            shared_net_fc_layer_list,
            shared_net_activation_func_list,
            shared_net_embedding_dim
        )

        if self.sequential_model == "lstm":
            self.seq_model = RNN_Base(
                input_dim=shared_net_embedding_dim,
                rnn_hidden_dim=shared_net_embedding_dim,
                rnn_num_layers=2,
                rnn_type="LSTM"
            )
        elif self.sequential_model == "gru":
            self.seq_model = RNN_Base(
                input_dim=shared_net_embedding_dim,
                rnn_hidden_dim=shared_net_embedding_dim,
                rnn_num_layers=2,
                rnn_type="GRU"
            )
        elif self.sequential_model == "gpt":
            self.seq_model = ContinuousGPT(
                input_dim=shared_net_embedding_dim,
                output_dim=shared_net_embedding_dim,
                max_seq_length=12,
            )
        else:
            self.seq_model = None


        if action_distribution == "deterministic":
            self.action_decoder = DeterministicActionDecoder(
                input_dim=shared_net_embedding_dim,
                action_dim=3 + rot_dim + 1
            )
        elif action_distribution == "gaussian":
            self.action_decoder = GaussianActionDecoder(
                input_dim=shared_net_embedding_dim,
                action_dim=3 + rot_dim + 1,
                use_low_noise_eval=use_low_noise_eval
            )
        elif action_distribution == "mixture":
            self.action_decoder = GaussianMixtureActionDecoder(
                input_dim=shared_net_embedding_dim,
                action_dim=3 + rot_dim + 1,
                use_low_noise_eval=use_low_noise_eval
            )

        # Pixel shift and random crops
        if use_shift_augmentation:
            self.shift_aug = RandomShiftsAug(pad=6)

    def forward(self, obs_dict):
        '''
        One forward pass of the network
        :param obs_dict: A dictionary containing the inputs to the network
                        {'key1' : <torch.tensor batch x seq x sensor_dim>,
                         'key2' : <torch.tensor>
                                    .
                                    .
                                    .
                        }
        '''
        rgb_tensor = obs_dict['rgb']
        depth_tensor = obs_dict['depth']
        ee_tensor = obs_dict['ee']
        gripper_tensor = obs_dict['gripper']

        n, l = rgb_tensor.shape[0], rgb_tensor.shape[1]

        # Pixel shift augmentation
        if self.use_shift_augmentation and self.training:
            rgb_tensor = self.shift_aug(
                rgb_tensor.reshape(-1, *rgb_tensor.shape[2:])
            )
            rgb_tensor = rgb_tensor.reshape(n, l, *rgb_tensor.shape[1:])

            if self.use_depth:
                depth_tensor = self.shift_aug(
                    depth_tensor.reshape(-1, *depth_tensor.shape[2:])
                )
                depth_tensor = depth_tensor.reshape(n, l, *depth_tensor.shape[1:])

        rgb_enc = self.rgb_encoder(
            rgb_tensor.reshape(-1, *rgb_tensor.shape[2:])
        )
        if self.use_spatial_encoding:
            shared_enc = self.spatial_softmax(self.kp_filter(rgb_enc))
        else:
            shared_enc = self.avg_pool(rgb_enc)
            shared_enc = shared_enc.reshape(*shared_enc.shape[:2])

        if self.use_depth:
            depth_enc = self.depth_encoder(depth_tensor.reshape(-1, *depth_tensor.shape[2:]))
            if self.use_spatial_encoding:
                depth_enc = self.spatial_softmax(self.kp_filter(depth_enc))
            else:
                depth_enc = self.avg_pool(depth_enc)
                depth_enc = depth_enc.reshape(*depth_enc.shape[:2])

            shared_enc = torch.cat([shared_enc, depth_enc], dim=-1)

        if self.use_ee:
            ee_enc = self.ee_net(ee_tensor.reshape(-1, *ee_tensor.shape[2:]))
            shared_enc = torch.cat([shared_enc, ee_enc], dim=-1)

        if self.use_gripper:
            gripper_enc = self.gripper_net(gripper_tensor.reshape(-1, *gripper_tensor.shape[2:]))
            shared_enc = torch.cat([shared_enc, gripper_enc], dim=-1)

        # Pass it through the FC layers
        embedding = self.shared_net(shared_enc)
        embedding = embedding.reshape(n, l, *embedding.shape[1:])

        if self.sequential_model == "lstm":
            embedding = self.seq_model(
                embedding,
                rnn_init_state=None,
                return_state=False
            )
        elif self.sequential_model == "gru":
            embedding = self.seq_model(
                embedding,
                rnn_init_state=None,
                return_state=False
            )
        elif self.sequential_model == "gpt":
            embedding = self.seq_model(embedding)

        a, pa_o = self.action_decoder(embedding)

        # Normalize
        if self.rotation_representation == "quat":
            a[3:6] = F.normalize(a[3:6], p=2, dim=-1)
        return a, pa_o

class MultimodalBCAnyObs(nn.Module):
    '''
    Another monstrosity of various model options to experiment conveniently.

    Sensor options are based on included keys, and initialization + shapes are chosen
    based on an example observation (at training time) or saved obs_key_shapes_dict
    (at control time).

    '''
    def __init__(
        self,
        rotation_representation,
        shared_net_fc_layer_list,
        shared_net_activation_func_list,
        shared_net_embedding_dim,
        action_distribution,
        n_kps,
        n_frame_stacks,
        use_shift_augmentation,
        use_spatial_encoding,
        sequential_model,
        obs_key_list,
        obs_key_shapes_dict,
        state_fc_layer_list,
        state_activation_func_list,
        state_embedding_dims,
        freeze_resnet,
        use_input_coord_conv,
        use_low_noise_eval,
        act_dim
    ):
        super().__init__()
        self.use_shift_augmentation = use_shift_augmentation
        self.use_spatial_encoding = use_spatial_encoding
        self.obs_key_list = obs_key_list
        self.sequential_model = sequential_model
        self.rotation_representation = rotation_representation
        self.n_frame_stacks = n_frame_stacks
        self.n_kps = n_kps
        fc_inp_size = 0

        # sort through all observation types
        self.img_encoder_names = []
        self.state_encoder_names = []

        for ok, v in obs_key_shapes_dict.items():
            if ok in IMG_KEYS:
                name = self.get_img_encoder_name(ok)
                self.img_encoder_names.append(name)
                setattr(self, name, ResNet18Conv(
                    pretrained=True,
                    input_channel=v[0] * (n_frame_stacks + 1),
                    input_coord_conv=use_input_coord_conv
                ))
                if freeze_resnet:
                    for param in getattr(self, name).parameters():
                        param.requires_grad = False

            else:
                state_dim = v[0]
                name = self.get_state_encoder_name(ok)
                self.state_encoder_names.append(name)
                setattr(self, name, build_fc_modules(
                    (state_dim) * (n_frame_stacks + 1),
                    state_fc_layer_list,
                    state_activation_func_list,
                    state_embedding_dims
                ))
                fc_inp_size += state_embedding_dims

        # Spatial encoder
        # Reduce channels to amount of keypoints
        if use_spatial_encoding:
            self.kp_filter = torch.nn.Conv2d(512, n_kps, kernel_size=1)
            self.spatial_softmax = SpatialSoftArgmax()
            fc_inp_size += len(self.img_encoder_names) * 2 * n_kps
        else:
            self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
            fc_inp_size += len(self.img_encoder_names) * 512

        self.shared_net = build_fc_modules(
            fc_inp_size,
            shared_net_fc_layer_list,
            shared_net_activation_func_list,
            shared_net_embedding_dim
        )

        # Sequential models
        if self.sequential_model == "lstm":
            self.seq_model = RNN_Base(
                input_dim=shared_net_embedding_dim,
                rnn_hidden_dim=shared_net_embedding_dim,
                rnn_num_layers=2,
                rnn_type="LSTM"
            )
        elif self.sequential_model == "gru":
            self.seq_model = RNN_Base(
                input_dim=shared_net_embedding_dim,
                rnn_hidden_dim=shared_net_embedding_dim,
                rnn_num_layers=2,
                rnn_type="GRU"
            )
        elif self.sequential_model == "gpt":
            self.seq_model = ContinuousGPT(
                input_dim=shared_net_embedding_dim,
                output_dim=shared_net_embedding_dim,
                max_seq_length=12,
            )
        else:
            self.seq_model = None

        # Action distribution
        if action_distribution == "deterministic":
            self.action_decoder = DeterministicActionDecoder(
                input_dim=shared_net_embedding_dim,
                action_dim=act_dim
            )
        elif action_distribution == "gaussian":
            self.action_decoder = GaussianActionDecoder(
                input_dim=shared_net_embedding_dim,
                action_dim=act_dim,
                use_low_noise_eval=use_low_noise_eval
            )
        elif action_distribution == "mixture":
            self.action_decoder = GaussianMixtureActionDecoder(
                input_dim=shared_net_embedding_dim,
                action_dim=act_dim,
                use_low_noise_eval=use_low_noise_eval
            )

        # Pixel shift and random crops
        if use_shift_augmentation:
            self.shift_aug = RandomShiftsAug(pad=6)

    def get_img_encoder_name(self, key):
        return self.get_encoder_name('img', key)

    def get_state_encoder_name(self, key):
        return self.get_encoder_name('state', key)

    def get_encoder_name(self, mode, key):
        return f'{mode}_encoder_{key}'

    def forward(self, obs_dict):
        '''
        One forward pass of the network
        :param obs_dict: A dictionary containing the inputs to the network
                        {'key1' : <torch.tensor batch x seq x sensor_dim>,
                         'key2' : <torch.tensor>
                                    .
                                    .
                                    .
                        }
        '''
        # Pixel shift augmentation
        shared_enc = None
        n, l = None, None
        for ok in self.obs_key_list:
            tensor = obs_dict[ok]
            if n is None:
                n, l = tensor.shape[0], tensor.shape[1]
            if ok in IMG_KEYS:
                if self.use_shift_augmentation and self.training:
                    tensor = self.shift_aug(
                        tensor.reshape(-1, tensor.shape[2:])
                    )
                    tensor = tensor.reshape(n, l, *tensor.shape[1:])

                enc = getattr(self, self.get_img_encoder_name(ok))(
                    tensor.reshape(-1, *tensor.shape[2:])
                )

                if self.use_spatial_encoding:
                    enc = self.spatial_softmax(self.kp_filter(enc))
                else:
                    enc = self.avg_pool(enc)
                    enc = enc.reshape(*enc.shape[:2])

            else:
                enc = getattr(self, self.get_state_encoder_name(ok))(tensor.reshape(-1, *tensor.shape[2:]))

            shared_enc = torch.cat([shared_enc, enc], dim=-1) if shared_enc is not None else enc

        # Pass it through the FC layers
        embedding = self.shared_net(shared_enc)
        embedding = embedding.reshape(n, l, *embedding.shape[1:])

        if self.sequential_model in {"lstm", "gru"}:
            embedding = self.seq_model(
                embedding,
                rnn_init_state=None,
                return_state=False
            )
        elif self.sequential_model == "gpt":
            embedding = self.seq_model(embedding)

        a, pa_o = self.action_decoder(embedding)

        # Normalize
        if self.rotation_representation == "quat":
            a[3:6] = F.normalize(a[3:6], p=2, dim=-1)
        return a, pa_o