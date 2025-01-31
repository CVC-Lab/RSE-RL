ʵ
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-0-gc256c071bb28��
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:0*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:0@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@H* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@H*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:H*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:H*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:H*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:H*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:H*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
�	�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�,B�, B�,
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
h

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
�
0
1
2
3
4
5
"6
#7
(8
)9
/10
011
112
213
;14
<15
A16
B17
v
0
1
2
3
4
5
"6
#7
(8
)9
/10
011
;12
<13
A14
B15
 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
$	variables
%trainable_variables
&regularization_losses
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
*	variables
+trainable_variables
,regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

/0
01
12
23

/0
01
 
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
3	variables
4trainable_variables
5regularization_losses
 
 
 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
7	variables
8trainable_variables
9regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
=	variables
>trainable_variables
?regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
 
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses

10
21
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_66768
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_67740
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_67804��
�
�
B__inference_encoder_layer_call_and_return_conditional_losses_66994

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 06
(conv2d_2_biasadd_readvariableop_resource:0A
'conv2d_3_conv2d_readvariableop_resource:0@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@H6
(conv2d_4_biasadd_readvariableop_resource:H9
+batch_normalization_readvariableop_resource:H;
-batch_normalization_readvariableop_1_resource:HJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:HL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H8
$dense_matmul_readvariableop_resource:
�	�4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:����������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hj
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������H�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:H*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:����������	�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: j
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:�����������	
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_67583O
5conv2d_kernel_regularizer_abs_readvariableop_resource:
identity��,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOpd
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5conv2d_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv2d_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp
�
�
&__inference_conv2d_layer_call_fn_67162

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_65717w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_1_layer_call_fn_67212

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65749w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
3__inference_batch_normalization_layer_call_fn_67414

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65673�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������H
 
_user_specified_nameinputs
�]
�
 __inference__wrapped_model_65620
input_1G
-encoder_conv2d_conv2d_readvariableop_resource:<
.encoder_conv2d_biasadd_readvariableop_resource:I
/encoder_conv2d_1_conv2d_readvariableop_resource: >
0encoder_conv2d_1_biasadd_readvariableop_resource: I
/encoder_conv2d_2_conv2d_readvariableop_resource: 0>
0encoder_conv2d_2_biasadd_readvariableop_resource:0I
/encoder_conv2d_3_conv2d_readvariableop_resource:0@>
0encoder_conv2d_3_biasadd_readvariableop_resource:@I
/encoder_conv2d_4_conv2d_readvariableop_resource:@H>
0encoder_conv2d_4_biasadd_readvariableop_resource:HA
3encoder_batch_normalization_readvariableop_resource:HC
5encoder_batch_normalization_readvariableop_1_resource:HR
Dencoder_batch_normalization_fusedbatchnormv3_readvariableop_resource:HT
Fencoder_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H@
,encoder_dense_matmul_readvariableop_resource:
�	�<
-encoder_dense_biasadd_readvariableop_resource:	�B
.encoder_dense_1_matmul_readvariableop_resource:
��>
/encoder_dense_1_biasadd_readvariableop_resource:	�
identity��;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp�=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�*encoder/batch_normalization/ReadVariableOp�,encoder/batch_normalization/ReadVariableOp_1�%encoder/conv2d/BiasAdd/ReadVariableOp�$encoder/conv2d/Conv2D/ReadVariableOp�'encoder/conv2d_1/BiasAdd/ReadVariableOp�&encoder/conv2d_1/Conv2D/ReadVariableOp�'encoder/conv2d_2/BiasAdd/ReadVariableOp�&encoder/conv2d_2/Conv2D/ReadVariableOp�'encoder/conv2d_3/BiasAdd/ReadVariableOp�&encoder/conv2d_3/Conv2D/ReadVariableOp�'encoder/conv2d_4/BiasAdd/ReadVariableOp�&encoder/conv2d_4/Conv2D/ReadVariableOp�$encoder/dense/BiasAdd/ReadVariableOp�#encoder/dense/MatMul/ReadVariableOp�&encoder/dense_1/BiasAdd/ReadVariableOp�%encoder/dense_1/MatMul/ReadVariableOp�
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
encoder/conv2d/Conv2DConv2Dinput_1,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������v
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:����������
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
encoder/conv2d_1/Conv2DConv2D!encoder/conv2d/Relu:activations:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
encoder/conv2d_2/Conv2DConv2D#encoder/conv2d_1/Relu:activations:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0z
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
encoder/conv2d_3/Conv2DConv2D#encoder/conv2d_2/Relu:activations:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
encoder/conv2d_4/Conv2DConv2D#encoder/conv2d_3/Relu:activations:0.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hz
encoder/conv2d_4/ReluRelu!encoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������H�
*encoder/batch_normalization/ReadVariableOpReadVariableOp3encoder_batch_normalization_readvariableop_resource*
_output_shapes
:H*
dtype0�
,encoder/batch_normalization/ReadVariableOp_1ReadVariableOp5encoder_batch_normalization_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpDencoder_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFencoder_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
,encoder/batch_normalization/FusedBatchNormV3FusedBatchNormV3#encoder/conv2d_4/Relu:activations:02encoder/batch_normalization/ReadVariableOp:value:04encoder/batch_normalization/ReadVariableOp_1:value:0Cencoder/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Eencoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( f
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
encoder/flatten/ReshapeReshape0encoder/batch_normalization/FusedBatchNormV3:y:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:����������	�
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0�
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
encoder/dense_1/MatMulMatMul encoder/dense/Relu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
encoder/dense_1/ReluRelu encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������r
IdentityIdentity"encoder/dense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp<^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp>^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^encoder/batch_normalization/ReadVariableOp-^encoder/batch_normalization/ReadVariableOp_1&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2z
;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp2~
=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_12X
*encoder/batch_normalization/ReadVariableOp*encoder/batch_normalization/ReadVariableOp2\
,encoder/batch_normalization/ReadVariableOp_1,encoder/batch_normalization/ReadVariableOp_12N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_67388

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������HX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������Hf
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67494

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_65996

inputs&
conv2d_65718:
conv2d_65720:(
conv2d_1_65750: 
conv2d_1_65752: (
conv2d_2_65782: 0
conv2d_2_65784:0(
conv2d_3_65814:0@
conv2d_3_65816:@(
conv2d_4_65846:@H
conv2d_4_65848:H'
batch_normalization_65869:H'
batch_normalization_65871:H'
batch_normalization_65873:H'
batch_normalization_65875:H
dense_65898:
�	�
dense_65900:	�!
dense_1_65915:
��
dense_1_65917:	�
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp� conv2d_4/StatefulPartitionedCall�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_65718conv2d_65720*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_65717�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_65750conv2d_1_65752*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65749�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_65782conv2d_2_65784*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65781�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_65814conv2d_3_65816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65813�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_65846conv2d_4_65848*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_65845�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_65869batch_normalization_65871batch_normalization_65873batch_normalization_65875*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65868�
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_65884�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_65898dense_65900*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_65897�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_65915dense_1_65917*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_65914d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_65718*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_65718*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_65750*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_65750*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_65782*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_65782*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_65814*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_65814*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_65846*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_65846*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65868

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_67188

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_67552

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_65914p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_3_layer_call_fn_67312

inputs!
unknown:0@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65813w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65673

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������H:H:H:H:H:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������H�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������H
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_65897

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_65717

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_67427

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65868w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66094

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65642

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������H:H:H:H:H:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������H�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������H
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_67623Q
7conv2d_2_kernel_regularizer_abs_readvariableop_resource: 0
identity��.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_2/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67476

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������H:H:H:H:H:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������H�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������H
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_65914

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_66526
input_1&
conv2d_66405:
conv2d_66407:(
conv2d_1_66410: 
conv2d_1_66412: (
conv2d_2_66415: 0
conv2d_2_66417:0(
conv2d_3_66420:0@
conv2d_3_66422:@(
conv2d_4_66425:@H
conv2d_4_66427:H'
batch_normalization_66430:H'
batch_normalization_66432:H'
batch_normalization_66434:H'
batch_normalization_66436:H
dense_66440:
�	�
dense_66442:	�!
dense_1_66445:
��
dense_1_66447:	�
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp� conv2d_4/StatefulPartitionedCall�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_66405conv2d_66407*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_65717�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_66410conv2d_1_66412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65749�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_66415conv2d_2_66417*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65781�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_66420conv2d_3_66422*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65813�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_66425conv2d_4_66427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_65845�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_66430batch_normalization_66432batch_normalization_66434batch_normalization_66436*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65868�
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_65884�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_66440dense_66442*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_65897�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_66445dense_1_66447*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_65914d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_66405*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_66405*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_66410*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_66410*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_66415*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_66415*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_66420*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_66420*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_66425*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_66425*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_67523

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������H:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
(__inference_conv2d_2_layer_call_fn_67262

inputs!
unknown: 0
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65781w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_67543

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
�
__inference_loss_fn_4_67663Q
7conv2d_4_kernel_regularizer_abs_readvariableop_resource:@H
identity��.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_4/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp
�
�
'__inference_encoder_layer_call_fn_66809

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:H

unknown_12:H

unknown_13:
�	�

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_65996p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_66322

inputs&
conv2d_66201:
conv2d_66203:(
conv2d_1_66206: 
conv2d_1_66208: (
conv2d_2_66211: 0
conv2d_2_66213:0(
conv2d_3_66216:0@
conv2d_3_66218:@(
conv2d_4_66221:@H
conv2d_4_66223:H'
batch_normalization_66226:H'
batch_normalization_66228:H'
batch_normalization_66230:H'
batch_normalization_66232:H
dense_66236:
�	�
dense_66238:	�!
dense_1_66241:
��
dense_1_66243:	�
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp� conv2d_4/StatefulPartitionedCall�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_66201conv2d_66203*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_65717�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_66206conv2d_1_66208*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65749�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_66211conv2d_2_66213*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65781�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_66216conv2d_3_66218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65813�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_66221conv2d_4_66223*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_65845�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_66226batch_normalization_66228batch_normalization_66230batch_normalization_66232*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66094�
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_65884�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_66236dense_66238*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_65897�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_66241dense_1_66243*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_65914d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_66201*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_66201*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_66206*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_66206*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_66211*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_66211*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_66216*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_66216*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_66221*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_66221*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_67643Q
7conv2d_3_kernel_regularizer_abs_readvariableop_resource:0@
identity��.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_3/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp
�
�
(__inference_conv2d_4_layer_call_fn_67362

inputs!
unknown:@H
	unknown_0:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_65845w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65749

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_66035
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:H

unknown_12:H

unknown_13:
�	�

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_65996p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_67138

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 06
(conv2d_2_biasadd_readvariableop_resource:0A
'conv2d_3_conv2d_readvariableop_resource:0@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@H6
(conv2d_4_biasadd_readvariableop_resource:H9
+batch_normalization_readvariableop_resource:H;
-batch_normalization_readvariableop_1_resource:HJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:HL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H8
$dense_matmul_readvariableop_resource:
�	�4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:����������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hj
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������H�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:H*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:����������	�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: j
IdentityIdentitydense_1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:�����������

NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67458

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������H:H:H:H:H:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������H�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������H
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_66402
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:H

unknown_12:H

unknown_13:
�	�

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_66322p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_66650
input_1&
conv2d_66529:
conv2d_66531:(
conv2d_1_66534: 
conv2d_1_66536: (
conv2d_2_66539: 0
conv2d_2_66541:0(
conv2d_3_66544:0@
conv2d_3_66546:@(
conv2d_4_66549:@H
conv2d_4_66551:H'
batch_normalization_66554:H'
batch_normalization_66556:H'
batch_normalization_66558:H'
batch_normalization_66560:H
dense_66564:
�	�
dense_66566:	�!
dense_1_66569:
��
dense_1_66571:	�
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp� conv2d_4/StatefulPartitionedCall�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_66529conv2d_66531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_65717�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_66534conv2d_1_66536*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65749�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_66539conv2d_2_66541*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65781�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_66544conv2d_3_66546*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65813�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_66549conv2d_4_66551*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_65845�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_66554batch_normalization_66556batch_normalization_66558batch_normalization_66560*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66094�
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_65884�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_66564dense_66566*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_65897�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_66569dense_1_66571*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_65914d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_66529*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_66529*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: f
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_66534*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_66534*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_66539*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_66539*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_66544*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_66544*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_66549*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_66549*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
3__inference_batch_normalization_layer_call_fn_67440

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_66094w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_67238

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� f
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_67603Q
7conv2d_1_kernel_regularizer_abs_readvariableop_resource: 
identity��.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65781

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������0f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������0�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_65845

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_4/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������HX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������Hf
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65813

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67512

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�-
�
__inference__traced_save_67740
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::: : : 0:0:0@:@:@H:H:H:H:H:H:
�	�:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@:,	(
&
_output_shapes
:@H: 


_output_shapes
:H: 

_output_shapes
:H: 

_output_shapes
:H: 

_output_shapes
:H: 

_output_shapes
:H:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: 
�	
�
3__inference_batch_normalization_layer_call_fn_67401

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_65642�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������H
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_67532

inputs
unknown:
�	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_65897p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_66850

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:H

unknown_12:H

unknown_13:
�	�

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_66322p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_66768
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:H

unknown_12:H

unknown_13:
�	�

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_65620p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_67338

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_65884

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������H:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�I
�
!__inference__traced_restore_67804
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel: .
 assignvariableop_3_conv2d_1_bias: <
"assignvariableop_4_conv2d_2_kernel: 0.
 assignvariableop_5_conv2d_2_bias:0<
"assignvariableop_6_conv2d_3_kernel:0@.
 assignvariableop_7_conv2d_3_bias:@<
"assignvariableop_8_conv2d_4_kernel:@H.
 assignvariableop_9_conv2d_4_bias:H;
-assignvariableop_10_batch_normalization_gamma:H:
,assignvariableop_11_batch_normalization_beta:HA
3assignvariableop_12_batch_normalization_moving_mean:HE
7assignvariableop_13_batch_normalization_moving_variance:H4
 assignvariableop_14_dense_kernel:
�	�-
assignvariableop_15_dense_bias:	�6
"assignvariableop_16_dense_1_kernel:
��/
 assignvariableop_17_dense_1_bias:	�
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_batch_normalization_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp,assignvariableop_11_batch_normalization_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp3assignvariableop_12_batch_normalization_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_67563

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_67288

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������0f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������0�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_67517

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_65884a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������H:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������<
dense_11
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api

signatures
y__call__
*z&call_and_return_all_conditional_losses
{_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0
1
2
3
4
5
"6
#7
(8
)9
/10
011
112
213
;14
<15
A16
B17"
trackable_list_wrapper
�
0
1
2
3
4
5
"6
#7
(8
)9
/10
011
;12
<13
A14
B15"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
y__call__
{_default_save_signature
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
):' 02conv2d_2/kernel
:02conv2d_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'0@2conv2d_3/kernel
:@2conv2d_3/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
$	variables
%trainable_variables
&regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'@H2conv2d_4/kernel
:H2conv2d_4/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
*	variables
+trainable_variables
,regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%H2batch_normalization/gamma
&:$H2batch_normalization/beta
/:-H (2batch_normalization/moving_mean
3:1H (2#batch_normalization/moving_variance
<
/0
01
12
23"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
3	variables
4trainable_variables
5regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
7	variables
8trainable_variables
9regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :
�	�2dense/kernel
:�2
dense/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
=	variables
>trainable_variables
?regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 
��2dense_1/kernel
:�2dense_1/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
'__inference_encoder_layer_call_fn_66035
'__inference_encoder_layer_call_fn_66809
'__inference_encoder_layer_call_fn_66850
'__inference_encoder_layer_call_fn_66402�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_encoder_layer_call_and_return_conditional_losses_66994
B__inference_encoder_layer_call_and_return_conditional_losses_67138
B__inference_encoder_layer_call_and_return_conditional_losses_66526
B__inference_encoder_layer_call_and_return_conditional_losses_66650�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_65620input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_conv2d_layer_call_fn_67162�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_conv2d_layer_call_and_return_conditional_losses_67188�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv2d_1_layer_call_fn_67212�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_67238�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv2d_2_layer_call_fn_67262�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_67288�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv2d_3_layer_call_fn_67312�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_67338�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv2d_4_layer_call_fn_67362�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_67388�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_batch_normalization_layer_call_fn_67401
3__inference_batch_normalization_layer_call_fn_67414
3__inference_batch_normalization_layer_call_fn_67427
3__inference_batch_normalization_layer_call_fn_67440�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67458
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67476
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67494
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67512�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_flatten_layer_call_fn_67517�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_flatten_layer_call_and_return_conditional_losses_67523�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_67532�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_layer_call_and_return_conditional_losses_67543�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_1_layer_call_fn_67552�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_67563�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_67583�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_67603�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_67623�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_67643�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_67663�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
#__inference_signature_wrapper_66768input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_65620�"#()/012;<AB8�5
.�+
)�&
input_1���������
� "2�/
-
dense_1"�
dense_1�����������
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67458�/012M�J
C�@
:�7
inputs+���������������������������H
p 
� "?�<
5�2
0+���������������������������H
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67476�/012M�J
C�@
:�7
inputs+���������������������������H
p
� "?�<
5�2
0+���������������������������H
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67494r/012;�8
1�.
(�%
inputs���������H
p 
� "-�*
#� 
0���������H
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67512r/012;�8
1�.
(�%
inputs���������H
p
� "-�*
#� 
0���������H
� �
3__inference_batch_normalization_layer_call_fn_67401�/012M�J
C�@
:�7
inputs+���������������������������H
p 
� "2�/+���������������������������H�
3__inference_batch_normalization_layer_call_fn_67414�/012M�J
C�@
:�7
inputs+���������������������������H
p
� "2�/+���������������������������H�
3__inference_batch_normalization_layer_call_fn_67427e/012;�8
1�.
(�%
inputs���������H
p 
� " ����������H�
3__inference_batch_normalization_layer_call_fn_67440e/012;�8
1�.
(�%
inputs���������H
p
� " ����������H�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_67238l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
(__inference_conv2d_1_layer_call_fn_67212_7�4
-�*
(�%
inputs���������
� " ���������� �
C__inference_conv2d_2_layer_call_and_return_conditional_losses_67288l7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������0
� �
(__inference_conv2d_2_layer_call_fn_67262_7�4
-�*
(�%
inputs��������� 
� " ����������0�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_67338l"#7�4
-�*
(�%
inputs���������0
� "-�*
#� 
0���������@
� �
(__inference_conv2d_3_layer_call_fn_67312_"#7�4
-�*
(�%
inputs���������0
� " ����������@�
C__inference_conv2d_4_layer_call_and_return_conditional_losses_67388l()7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������H
� �
(__inference_conv2d_4_layer_call_fn_67362_()7�4
-�*
(�%
inputs���������@
� " ����������H�
A__inference_conv2d_layer_call_and_return_conditional_losses_67188l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
&__inference_conv2d_layer_call_fn_67162_7�4
-�*
(�%
inputs���������
� " �����������
B__inference_dense_1_layer_call_and_return_conditional_losses_67563^AB0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dense_1_layer_call_fn_67552QAB0�-
&�#
!�
inputs����������
� "������������
@__inference_dense_layer_call_and_return_conditional_losses_67543^;<0�-
&�#
!�
inputs����������	
� "&�#
�
0����������
� z
%__inference_dense_layer_call_fn_67532Q;<0�-
&�#
!�
inputs����������	
� "������������
B__inference_encoder_layer_call_and_return_conditional_losses_66526~"#()/012;<AB@�=
6�3
)�&
input_1���������
p 

 
� "&�#
�
0����������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_66650~"#()/012;<AB@�=
6�3
)�&
input_1���������
p

 
� "&�#
�
0����������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_66994}"#()/012;<AB?�<
5�2
(�%
inputs���������
p 

 
� "&�#
�
0����������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_67138}"#()/012;<AB?�<
5�2
(�%
inputs���������
p

 
� "&�#
�
0����������
� �
'__inference_encoder_layer_call_fn_66035q"#()/012;<AB@�=
6�3
)�&
input_1���������
p 

 
� "������������
'__inference_encoder_layer_call_fn_66402q"#()/012;<AB@�=
6�3
)�&
input_1���������
p

 
� "������������
'__inference_encoder_layer_call_fn_66809p"#()/012;<AB?�<
5�2
(�%
inputs���������
p 

 
� "������������
'__inference_encoder_layer_call_fn_66850p"#()/012;<AB?�<
5�2
(�%
inputs���������
p

 
� "������������
B__inference_flatten_layer_call_and_return_conditional_losses_67523a7�4
-�*
(�%
inputs���������H
� "&�#
�
0����������	
� 
'__inference_flatten_layer_call_fn_67517T7�4
-�*
(�%
inputs���������H
� "�����������	:
__inference_loss_fn_0_67583�

� 
� "� :
__inference_loss_fn_1_67603�

� 
� "� :
__inference_loss_fn_2_67623�

� 
� "� :
__inference_loss_fn_3_67643"�

� 
� "� :
__inference_loss_fn_4_67663(�

� 
� "� �
#__inference_signature_wrapper_66768�"#()/012;<ABC�@
� 
9�6
4
input_1)�&
input_1���������"2�/
-
dense_1"�
dense_1����������