import torch
import math
import torch.nn as nn
from models.vq.resnet import Resnet1D
import clip
from utils.paramUtil import animo_kinematic_chain


class PositionalEncoding(nn.Module):
    def __init__(self, src_dim, embed_dim, dropout, max_len=100, hid_dim=512):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(src_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, embed_dim)
        self.relu = nn.ReLU()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embed_dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / embed_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, step=None):
        emb = self.linear2(self.relu(self.linear1(input)))
        emb = emb * math.sqrt(self.embed_dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class Spatial_Transformer(nn.Module):
    def __init__(self, transformer_layers,
                 transformer_latents,
                 transformer_ffsize,
                 transformer_heads,
                 transformer_dropout,
                 transformer_srcdim,
                 correspondence,
                 njoints,
                 activation="gelu"):
        super(Spatial_Transformer, self).__init__()
        self.correspondence = correspondence
        self.nparts = len(correspondence)
        self.njoints = njoints 
        self.num_layers = transformer_layers
        self.latent_dim = transformer_latents
        self.ff_size = transformer_ffsize
        self.num_heads = transformer_heads
        self.dropout = transformer_dropout
        self.src_dim = transformer_srcdim
        self.activation = activation

        self.joint_pos_encoder = PositionalEncoding(self.src_dim, self.latent_dim, self.dropout)
        spaceTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.spatialTransEncoder = nn.TransformerEncoder(spaceTransEncoderLayer,
                                                         num_layers=self.num_layers)

    def forward(self, x, attention_mask,offset=None):


        b, t, j,c= x.shape[0], x.shape[1], x.shape[2],x.shape[3]
        x = x.transpose(0, 2).reshape(j,t*b,c)  # J BT E
        encoding = self.joint_pos_encoder(x)
        final = self.spatialTransEncoder(encoding, mask=attention_mask)
        joints_emb = final[:30].reshape(30,t,b,-1).transpose(0,2) # B * nparts E * T
        return joints_emb
    
class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        self.spatial_transformer = Spatial_Transformer(transformer_layers=2,
                                                        transformer_latents=128,
                                                        transformer_ffsize=512,
                                                        transformer_heads=4,
                                                        transformer_dropout= 0.2,
                                                        transformer_srcdim=32,
                                                        correspondence=animo_kinematic_chain,
                                                        njoints=31
                                                        )
        self.root_joint_embed = nn.Linear(7, 32)
        self.other_joint_embed = nn.Linear(12, 32)
        self.contact_embed = nn.Linear(4, 32)
        self.film_layer=FiLMLayer()
        self.clip_model = self.load_and_freeze_clip()
        
    
    def load_and_freeze_clip(self,clip_version='ViT-B/32'):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
                clip_model)  
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    def get_transformer_matrix(self,part_list, njoints):
        nparts = len(part_list)
        matrix = torch.zeros([ njoints, njoints])

        for i in range(nparts):
            for j in part_list[i]:
                for k in part_list[i]:
                    matrix[j , k] = 1
            
        matrix[:, 0] = 1
        matrix[:, -1] = 1
        for p in range(njoints):
            matrix[p, p] = 1
        matrix = matrix.float().masked_fill(matrix == 0., float(-1e20)).masked_fill(matrix == 1., float(0.0))
        return matrix
    
    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def forward(self, species,gender, x):
        x = x.permute(0,2,1).float()
        B, T = x.shape[0], x.shape[1]
        root_feature = torch.cat((x[:,:,:4],x[:,:,265:268]),dim=2).unsqueeze(2)
        other_joint = torch.cat((x[:,:,4:265],x[:,:,268:355]),dim=2)
        position = other_joint[:,:,:87].reshape(B, T,29,3)
        rotation = other_joint[:,:,87:261].reshape(B, T,29,6)
        velocity = other_joint[:,:,261:].reshape(B, T,29,3)
        other_joint_feature = torch.cat((torch.cat((position,rotation),dim=3),velocity),dim=3)
        contact = x[:,:,355:].unsqueeze(2)
        root_feature = self.root_joint_embed(root_feature)
        other_joint_feature = self.other_joint_embed(other_joint_feature)
        contact_feature = self.contact_embed(contact)
        h = torch.cat((torch.cat((root_feature,other_joint_feature),dim=2),contact_feature),dim=2)
        attention_mask = self.get_transformer_matrix(animo_kinematic_chain, 31).to(x.device)
        text_encoded = self.encode_text([f"{a} {b}" for a, b in zip(species, gender)])
        h = self.spatial_transformer(h,attention_mask.to(x.device)).reshape(B,T,30*128)
        h = h.permute(0,2,1)
        out = self.model(h) 
        out = self.film_layer(out, text_encoded)
        return out
    



class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        self.film_layer=FiLMLayer()
    

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)
      
      
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim=512, condition_dim=512):
        super(FiLMLayer, self).__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.to_cond = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, feature_tensor, condition_vector):
        assert feature_tensor.size(0) == condition_vector.size(0), "Batch sizes do not match"
        assert feature_tensor.size(1) == self.feature_dim, "Feature dimensions do not match"
        assert condition_vector.size(1) == self.condition_dim, "Condition dimensions do not match"
        
        ret = self.to_cond(condition_vector)
        gamma, beta = ret.chunk(2, dim=-1)
    
        gamma = gamma.unsqueeze(-1)  # (batch_size, feature_dim, 1)
        beta = beta.unsqueeze(-1)    # (batch_size, feature_dim, 1)
        
        return feature_tensor * gamma + beta