net = dict(
    type='ResHWLane', #'RESANet',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    fea_stride=8,
  #  in_channels=[64, 128, 256, -1],
)

mfia = dict(
    type='MFIA',
    alpha=2.0,
    iter=2,
    input_channel=128,
    conv_stride=9,
)

decoder = 'PlainDecoder'        

trainer = dict(
    type='Lane'
)

evaluator = dict(
    type='VILane',        
)

# optimizer = dict(
#   type='AdamW',
#   lr=0.0015,
#   weight_decay=0.03,
#   betas = (0.9, 0.999)
# )
optimizer = dict(
 type='SGD',
 lr=0.03,
 weight_decay=1e-4,
 momentum=0.9
)
hw_type = 1
epochs = 24
batch_size = 8
t = 0.5
total_iter = (8000 // batch_size) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

loss_type = 'cross_entropy'
seg_loss_weight = 2.4
eval_ep = 2
save_ep = 2

bg_weight = 0.3

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)
h = 1080
w = 1920
img_height = 368
img_width = 640
cut_height = 160
depth = 2
kd_epoch = 12
dataset_path = './data/VIL100'
dataset = dict(
    train=dict(
        type='VILane',
        img_path=dataset_path,
        data_list='train.txt',
    ),
    val=dict(
        type='VILane',
        img_path=dataset_path,
        data_list='test.txt',
    ),
    test=dict(
        type='VILane',
        img_path=dataset_path,
        data_list='test.txt',
    )
)

workers = 12
num_classes = 8 + 1
ignore_label = 255
log_interval = 500
