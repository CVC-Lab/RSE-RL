��
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
�
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
�
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
�
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: 0*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:0*
dtype0
�
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:0@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0
�
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@H* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:@H*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:H*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:H*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:H*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:H*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:H*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
��*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:�*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
��*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
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
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

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
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_input_3Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
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
#__inference_signature_wrapper_95506
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_96478
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
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
!__inference__traced_restore_96542��
�
�
__inference_loss_fn_0_96321Q
7conv2d_5_kernel_regularizer_abs_readvariableop_resource:
identity��.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_5_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_5_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_5/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp
�
�
'__inference_dense_4_layer_call_fn_96270

inputs
unknown:
��
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
GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_94635p
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
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_96361Q
7conv2d_7_kernel_regularizer_abs_readvariableop_resource: 0
identity��.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_7_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_7_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_7/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp
�

�
B__inference_dense_4_layer_call_and_return_conditional_losses_96281

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_94622

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������H:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94411

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
�
C__inference_conv2d_9_layer_call_and_return_conditional_losses_94583

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
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
:���������HX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������Hf
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_conv2d_8_layer_call_fn_96050

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
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_94551w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
(__inference_conv2d_7_layer_call_fn_96000

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
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_94519w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_conv2d_5_layer_call_fn_95900

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
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_94455w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_4_96401Q
7conv2d_9_kernel_regularizer_abs_readvariableop_resource:@H
identity��.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_9_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_9_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_9/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_95060

inputs(
conv2d_5_94939:
conv2d_5_94941:(
conv2d_6_94944: 
conv2d_6_94946: (
conv2d_7_94949: 0
conv2d_7_94951:0(
conv2d_8_94954:0@
conv2d_8_94956:@(
conv2d_9_94959:@H
conv2d_9_94961:H)
batch_normalization_1_94964:H)
batch_normalization_1_94966:H)
batch_normalization_1_94968:H)
batch_normalization_1_94970:H!
dense_4_94974:
��
dense_4_94976:	�!
dense_5_94979:
��
dense_5_94981:	�
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp� conv2d_7/StatefulPartitionedCall�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp� conv2d_8/StatefulPartitionedCall�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp� conv2d_9/StatefulPartitionedCall�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_94939conv2d_5_94941*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_94455�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_94944conv2d_6_94946*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_94487�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_94949conv2d_7_94951*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_94519�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_94954conv2d_8_94956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_94551�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_94959conv2d_9_94961*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_94583�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_1_94964batch_normalization_1_94966batch_normalization_1_94968batch_normalization_1_94970*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94832�
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_94622�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_94974dense_4_94976*
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
GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_94635�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_94979dense_5_94981*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_94652f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_94939*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_94939*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_94944*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_94944*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_94949*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_94949*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_94954*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_94954*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_94959*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_94959*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96214

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
�^
�
 __inference__wrapped_model_94358
input_3I
/encoder_conv2d_5_conv2d_readvariableop_resource:>
0encoder_conv2d_5_biasadd_readvariableop_resource:I
/encoder_conv2d_6_conv2d_readvariableop_resource: >
0encoder_conv2d_6_biasadd_readvariableop_resource: I
/encoder_conv2d_7_conv2d_readvariableop_resource: 0>
0encoder_conv2d_7_biasadd_readvariableop_resource:0I
/encoder_conv2d_8_conv2d_readvariableop_resource:0@>
0encoder_conv2d_8_biasadd_readvariableop_resource:@I
/encoder_conv2d_9_conv2d_readvariableop_resource:@H>
0encoder_conv2d_9_biasadd_readvariableop_resource:HC
5encoder_batch_normalization_1_readvariableop_resource:HE
7encoder_batch_normalization_1_readvariableop_1_resource:HT
Fencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:HV
Hencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:HB
.encoder_dense_4_matmul_readvariableop_resource:
��>
/encoder_dense_4_biasadd_readvariableop_resource:	�B
.encoder_dense_5_matmul_readvariableop_resource:
��>
/encoder_dense_5_biasadd_readvariableop_resource:	�
identity��=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�,encoder/batch_normalization_1/ReadVariableOp�.encoder/batch_normalization_1/ReadVariableOp_1�'encoder/conv2d_5/BiasAdd/ReadVariableOp�&encoder/conv2d_5/Conv2D/ReadVariableOp�'encoder/conv2d_6/BiasAdd/ReadVariableOp�&encoder/conv2d_6/Conv2D/ReadVariableOp�'encoder/conv2d_7/BiasAdd/ReadVariableOp�&encoder/conv2d_7/Conv2D/ReadVariableOp�'encoder/conv2d_8/BiasAdd/ReadVariableOp�&encoder/conv2d_8/Conv2D/ReadVariableOp�'encoder/conv2d_9/BiasAdd/ReadVariableOp�&encoder/conv2d_9/Conv2D/ReadVariableOp�&encoder/dense_4/BiasAdd/ReadVariableOp�%encoder/dense_4/MatMul/ReadVariableOp�&encoder/dense_5/BiasAdd/ReadVariableOp�%encoder/dense_5/MatMul/ReadVariableOp�
&encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
encoder/conv2d_5/Conv2DConv2Dinput_3.encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
'encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder/conv2d_5/BiasAddBiasAdd encoder/conv2d_5/Conv2D:output:0/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������z
encoder/conv2d_5/ReluRelu!encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:����������
&encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
encoder/conv2d_6/Conv2DConv2D#encoder/conv2d_5/Relu:activations:0.encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
'encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
encoder/conv2d_6/BiasAddBiasAdd encoder/conv2d_6/Conv2D:output:0/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
encoder/conv2d_6/ReluRelu!encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
&encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
encoder/conv2d_7/Conv2DConv2D#encoder/conv2d_6/Relu:activations:0.encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
'encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
encoder/conv2d_7/BiasAddBiasAdd encoder/conv2d_7/Conv2D:output:0/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0z
encoder/conv2d_7/ReluRelu!encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
&encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
encoder/conv2d_8/Conv2DConv2D#encoder/conv2d_7/Relu:activations:0.encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2d_8/BiasAddBiasAdd encoder/conv2d_8/Conv2D:output:0/encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
encoder/conv2d_8/ReluRelu!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
&encoder/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
encoder/conv2d_9/Conv2DConv2D#encoder/conv2d_8/Relu:activations:0.encoder/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
'encoder/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
encoder/conv2d_9/BiasAddBiasAdd encoder/conv2d_9/Conv2D:output:0/encoder/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hz
encoder/conv2d_9/ReluRelu!encoder/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������H�
,encoder/batch_normalization_1/ReadVariableOpReadVariableOp5encoder_batch_normalization_1_readvariableop_resource*
_output_shapes
:H*
dtype0�
.encoder/batch_normalization_1/ReadVariableOp_1ReadVariableOp7encoder_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpFencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
.encoder/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#encoder/conv2d_9/Relu:activations:04encoder/batch_normalization_1/ReadVariableOp:value:06encoder/batch_normalization_1/ReadVariableOp_1:value:0Eencoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Gencoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( h
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 
  �
encoder/flatten_1/ReshapeReshape2encoder/batch_normalization_1/FusedBatchNormV3:y:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:�����������
%encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
encoder/dense_4/MatMulMatMul"encoder/flatten_1/Reshape:output:0-encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/dense_4/BiasAddBiasAdd encoder/dense_4/MatMul:product:0.encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
encoder/dense_4/ReluRelu encoder/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%encoder/dense_5/MatMul/ReadVariableOpReadVariableOp.encoder_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
encoder/dense_5/MatMulMatMul"encoder/dense_4/Relu:activations:0-encoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&encoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/dense_5/BiasAddBiasAdd encoder/dense_5/MatMul:product:0.encoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
encoder/dense_5/ReluRelu encoder/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������r
IdentityIdentity"encoder/dense_5/Relu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp>^encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^encoder/batch_normalization_1/ReadVariableOp/^encoder/batch_normalization_1/ReadVariableOp_1(^encoder/conv2d_5/BiasAdd/ReadVariableOp'^encoder/conv2d_5/Conv2D/ReadVariableOp(^encoder/conv2d_6/BiasAdd/ReadVariableOp'^encoder/conv2d_6/Conv2D/ReadVariableOp(^encoder/conv2d_7/BiasAdd/ReadVariableOp'^encoder/conv2d_7/Conv2D/ReadVariableOp(^encoder/conv2d_8/BiasAdd/ReadVariableOp'^encoder/conv2d_8/Conv2D/ReadVariableOp(^encoder/conv2d_9/BiasAdd/ReadVariableOp'^encoder/conv2d_9/Conv2D/ReadVariableOp'^encoder/dense_4/BiasAdd/ReadVariableOp&^encoder/dense_4/MatMul/ReadVariableOp'^encoder/dense_5/BiasAdd/ReadVariableOp&^encoder/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2~
=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12\
,encoder/batch_normalization_1/ReadVariableOp,encoder/batch_normalization_1/ReadVariableOp2`
.encoder/batch_normalization_1/ReadVariableOp_1.encoder/batch_normalization_1/ReadVariableOp_12R
'encoder/conv2d_5/BiasAdd/ReadVariableOp'encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&encoder/conv2d_5/Conv2D/ReadVariableOp&encoder/conv2d_5/Conv2D/ReadVariableOp2R
'encoder/conv2d_6/BiasAdd/ReadVariableOp'encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&encoder/conv2d_6/Conv2D/ReadVariableOp&encoder/conv2d_6/Conv2D/ReadVariableOp2R
'encoder/conv2d_7/BiasAdd/ReadVariableOp'encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&encoder/conv2d_7/Conv2D/ReadVariableOp&encoder/conv2d_7/Conv2D/ReadVariableOp2R
'encoder/conv2d_8/BiasAdd/ReadVariableOp'encoder/conv2d_8/BiasAdd/ReadVariableOp2P
&encoder/conv2d_8/Conv2D/ReadVariableOp&encoder/conv2d_8/Conv2D/ReadVariableOp2R
'encoder/conv2d_9/BiasAdd/ReadVariableOp'encoder/conv2d_9/BiasAdd/ReadVariableOp2P
&encoder/conv2d_9/Conv2D/ReadVariableOp&encoder/conv2d_9/Conv2D/ReadVariableOp2P
&encoder/dense_4/BiasAdd/ReadVariableOp&encoder/dense_4/BiasAdd/ReadVariableOp2N
%encoder/dense_4/MatMul/ReadVariableOp%encoder/dense_4/MatMul/ReadVariableOp2P
&encoder/dense_5/BiasAdd/ReadVariableOp&encoder/dense_5/BiasAdd/ReadVariableOp2N
%encoder/dense_5/MatMul/ReadVariableOp%encoder/dense_5/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
__inference_loss_fn_1_96341Q
7conv2d_6_kernel_regularizer_abs_readvariableop_resource: 
identity��.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_6/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp
�
�
'__inference_encoder_layer_call_fn_95588

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
��

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
B__inference_encoder_layer_call_and_return_conditional_losses_95060p
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
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_95264
input_3(
conv2d_5_95143:
conv2d_5_95145:(
conv2d_6_95148: 
conv2d_6_95150: (
conv2d_7_95153: 0
conv2d_7_95155:0(
conv2d_8_95158:0@
conv2d_8_95160:@(
conv2d_9_95163:@H
conv2d_9_95165:H)
batch_normalization_1_95168:H)
batch_normalization_1_95170:H)
batch_normalization_1_95172:H)
batch_normalization_1_95174:H!
dense_4_95178:
��
dense_4_95180:	�!
dense_5_95183:
��
dense_5_95185:	�
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp� conv2d_7/StatefulPartitionedCall�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp� conv2d_8/StatefulPartitionedCall�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp� conv2d_9/StatefulPartitionedCall�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_5_95143conv2d_5_95145*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_94455�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_95148conv2d_6_95150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_94487�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_95153conv2d_7_95155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_94519�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_95158conv2d_8_95160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_94551�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_95163conv2d_9_95165*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_94583�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_1_95168batch_normalization_1_95170batch_normalization_1_95172batch_normalization_1_95174*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94606�
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_94622�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_95178dense_4_95180*
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
GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_94635�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_95183dense_5_95185*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_94652f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_95143*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_95143*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_95148*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_95148*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_95153*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_95153*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_95158*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_95158*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_95163*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_95163*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_94487

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_96165

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
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94606w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
C__inference_conv2d_9_layer_call_and_return_conditional_losses_96126

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
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
:���������HX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������Hf
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_95926

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_94773
input_3!
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
��

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
B__inference_encoder_layer_call_and_return_conditional_losses_94734p
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
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_96026

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
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
:���������0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������0f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������0�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_94455

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
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
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94606

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
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96232

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
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������H�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_96261

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 
  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������H:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94380

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
�
�
(__inference_conv2d_9_layer_call_fn_96100

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
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_94583w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_94519

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
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
:���������0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������0f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������0�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
B__inference_encoder_layer_call_and_return_conditional_losses_95732

inputsA
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: 06
(conv2d_7_biasadd_readvariableop_resource:0A
'conv2d_8_conv2d_readvariableop_resource:0@6
(conv2d_8_biasadd_readvariableop_resource:@A
'conv2d_9_conv2d_readvariableop_resource:@H6
(conv2d_9_biasadd_readvariableop_resource:H;
-batch_normalization_1_readvariableop_resource:H=
/batch_normalization_1_readvariableop_1_resource:HL
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:H:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�:
&dense_5_matmul_readvariableop_resource:
��6
'dense_5_biasadd_readvariableop_resource:	�
identity��5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:����������
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/Conv2DConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_8/Conv2DConv2Dconv2d_7/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hj
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������H�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:H*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_9/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
is_training( `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 
  �
flatten_1/ReshapeReshape*batch_normalization_1/FusedBatchNormV3:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:�����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: j
IdentityIdentitydense_5/Relu:activations:0^NoOp*
T0*(
_output_shapes
:�����������	
NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_96381Q
7conv2d_8_kernel_regularizer_abs_readvariableop_resource:0@
identity��.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_8_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_8_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_8/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96250

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
7:���������H:H:H:H:H:*
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
:���������H�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�-
�
__inference__traced_save_96478
file_prefix.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
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
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
��:�:
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
��:!
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
�
�
'__inference_dense_5_layer_call_fn_96290

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
B__inference_dense_5_layer_call_and_return_conditional_losses_94652p
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
�J
�
!__inference__traced_restore_96542
file_prefix:
 assignvariableop_conv2d_5_kernel:.
 assignvariableop_1_conv2d_5_bias:<
"assignvariableop_2_conv2d_6_kernel: .
 assignvariableop_3_conv2d_6_bias: <
"assignvariableop_4_conv2d_7_kernel: 0.
 assignvariableop_5_conv2d_7_bias:0<
"assignvariableop_6_conv2d_8_kernel:0@.
 assignvariableop_7_conv2d_8_bias:@<
"assignvariableop_8_conv2d_9_kernel:@H.
 assignvariableop_9_conv2d_9_bias:H=
/assignvariableop_10_batch_normalization_1_gamma:H<
.assignvariableop_11_batch_normalization_1_beta:HC
5assignvariableop_12_batch_normalization_1_moving_mean:HG
9assignvariableop_13_batch_normalization_1_moving_variance:H6
"assignvariableop_14_dense_4_kernel:
��/
 assignvariableop_15_dense_4_bias:	�6
"assignvariableop_16_dense_5_kernel:
��/
 assignvariableop_17_dense_5_bias:	�
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
AssignVariableOpAssignVariableOp assignvariableop_conv2d_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_8_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_8_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_9_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_9_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_5_biasIdentity_17:output:0"/device:CPU:0*
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
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94832

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
7:���������H:H:H:H:H:*
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
:���������H�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�

�
B__inference_dense_5_layer_call_and_return_conditional_losses_96301

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
�
E
)__inference_flatten_1_layer_call_fn_96255

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_94622a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������H:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_95388
input_3(
conv2d_5_95267:
conv2d_5_95269:(
conv2d_6_95272: 
conv2d_6_95274: (
conv2d_7_95277: 0
conv2d_7_95279:0(
conv2d_8_95282:0@
conv2d_8_95284:@(
conv2d_9_95287:@H
conv2d_9_95289:H)
batch_normalization_1_95292:H)
batch_normalization_1_95294:H)
batch_normalization_1_95296:H)
batch_normalization_1_95298:H!
dense_4_95302:
��
dense_4_95304:	�!
dense_5_95307:
��
dense_5_95309:	�
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp� conv2d_7/StatefulPartitionedCall�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp� conv2d_8/StatefulPartitionedCall�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp� conv2d_9/StatefulPartitionedCall�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_5_95267conv2d_5_95269*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_94455�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_95272conv2d_6_95274*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_94487�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_95277conv2d_7_95279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_94519�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_95282conv2d_8_95284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_94551�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_95287conv2d_9_95289*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_94583�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_1_95292batch_normalization_1_95294batch_normalization_1_95296batch_normalization_1_95298*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94832�
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_94622�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_95302dense_4_95304*
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
GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_94635�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_95307dense_5_95309*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_94652f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_95267*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_95267*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_95272*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_95272*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_95277*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_95277*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_95282*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_95282*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_95287*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_95287*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
'__inference_encoder_layer_call_fn_95140
input_3!
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
��

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
B__inference_encoder_layer_call_and_return_conditional_losses_95060p
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
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_94551

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_95976

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
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
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_1_layer_call_fn_96139

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
GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94380�
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
�
�
5__inference_batch_normalization_1_layer_call_fn_96178

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
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94832w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_95547

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
��

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
B__inference_encoder_layer_call_and_return_conditional_losses_94734p
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
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
B__inference_dense_5_layer_call_and_return_conditional_losses_94652

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
Ͻ
�
B__inference_encoder_layer_call_and_return_conditional_losses_95876

inputsA
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: 06
(conv2d_7_biasadd_readvariableop_resource:0A
'conv2d_8_conv2d_readvariableop_resource:0@6
(conv2d_8_biasadd_readvariableop_resource:@A
'conv2d_9_conv2d_readvariableop_resource:@H6
(conv2d_9_biasadd_readvariableop_resource:H;
-batch_normalization_1_readvariableop_resource:H=
/batch_normalization_1_readvariableop_1_resource:HL
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:H:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�:
&dense_5_matmul_readvariableop_resource:
��6
'dense_5_biasadd_readvariableop_resource:	�
identity��$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:����������
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/Conv2DConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������0j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������0�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_8/Conv2DConv2Dconv2d_7/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hj
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������H�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:H*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_9/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������H:H:H:H:H:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 
  �
flatten_1/ReshapeReshape*batch_normalization_1/FusedBatchNormV3:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:�����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: j
IdentityIdentitydense_5/Relu:activations:0^NoOp*
T0*(
_output_shapes
:�����������

NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_96076

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
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
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96196

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

�
B__inference_dense_4_layer_call_and_return_conditional_losses_94635

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_95506
input_3!
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
��

unknown_14:	�

unknown_15:
��

unknown_16:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 __inference__wrapped_model_94358p
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
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_3
��
�
B__inference_encoder_layer_call_and_return_conditional_losses_94734

inputs(
conv2d_5_94456:
conv2d_5_94458:(
conv2d_6_94488: 
conv2d_6_94490: (
conv2d_7_94520: 0
conv2d_7_94522:0(
conv2d_8_94552:0@
conv2d_8_94554:@(
conv2d_9_94584:@H
conv2d_9_94586:H)
batch_normalization_1_94607:H)
batch_normalization_1_94609:H)
batch_normalization_1_94611:H)
batch_normalization_1_94613:H!
dense_4_94636:
��
dense_4_94638:	�!
dense_5_94653:
��
dense_5_94655:	�
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall�.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_5/kernel/Regularizer/Square/ReadVariableOp� conv2d_6/StatefulPartitionedCall�.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_6/kernel/Regularizer/Square/ReadVariableOp� conv2d_7/StatefulPartitionedCall�.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_7/kernel/Regularizer/Square/ReadVariableOp� conv2d_8/StatefulPartitionedCall�.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_8/kernel/Regularizer/Square/ReadVariableOp� conv2d_9/StatefulPartitionedCall�.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_9/kernel/Regularizer/Square/ReadVariableOp�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_94456conv2d_5_94458*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_94455�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_94488conv2d_6_94490*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_94487�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_94520conv2d_7_94522*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_94519�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_94552conv2d_8_94554*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_94551�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_94584conv2d_9_94586*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_94583�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_1_94607batch_normalization_1_94609batch_normalization_1_94611batch_normalization_1_94613*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94606�
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_94622�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_94636dense_4_94638*
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
GPU2*0J 8� *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_94635�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_94653dense_5_94655*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_94652f
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_94456*&
_output_shapes
:*
dtype0�
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_94456*&
_output_shapes
:*
dtype0�
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_94488*&
_output_shapes
: *
dtype0�
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_94488*&
_output_shapes
: *
dtype0�
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_94520*&
_output_shapes
: 0*
dtype0�
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_94520*&
_output_shapes
: 0*
dtype0�
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0|
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_94552*&
_output_shapes
:0@*
dtype0�
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_94552*&
_output_shapes
:0@*
dtype0�
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_94584*&
_output_shapes
:@H*
dtype0�
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             �
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_94584*&
_output_shapes
:@H*
dtype0�
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             �
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: �
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
5__inference_batch_normalization_1_layer_call_fn_96152

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
GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_94411�
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
(__inference_conv2d_6_layer_call_fn_95950

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
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_94487w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
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
input_38
serving_default_input_3:0���������<
dense_51
StatefulPartitionedCall:0����������tensorflow/serving/predict:�
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
):'2conv2d_5/kernel
:2conv2d_5/bias
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
):' 2conv2d_6/kernel
: 2conv2d_6/bias
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
):' 02conv2d_7/kernel
:02conv2d_7/bias
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
):'0@2conv2d_8/kernel
:@2conv2d_8/bias
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
):'@H2conv2d_9/kernel
:H2conv2d_9/bias
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
):'H2batch_normalization_1/gamma
(:&H2batch_normalization_1/beta
1:/H (2!batch_normalization_1/moving_mean
5:3H (2%batch_normalization_1/moving_variance
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
": 
��2dense_4/kernel
:�2dense_4/bias
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
��2dense_5/kernel
:�2dense_5/bias
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
'__inference_encoder_layer_call_fn_94773
'__inference_encoder_layer_call_fn_95547
'__inference_encoder_layer_call_fn_95588
'__inference_encoder_layer_call_fn_95140�
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
B__inference_encoder_layer_call_and_return_conditional_losses_95732
B__inference_encoder_layer_call_and_return_conditional_losses_95876
B__inference_encoder_layer_call_and_return_conditional_losses_95264
B__inference_encoder_layer_call_and_return_conditional_losses_95388�
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
 __inference__wrapped_model_94358input_3"�
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
(__inference_conv2d_5_layer_call_fn_95900�
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_95926�
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
(__inference_conv2d_6_layer_call_fn_95950�
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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_95976�
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
(__inference_conv2d_7_layer_call_fn_96000�
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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_96026�
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
(__inference_conv2d_8_layer_call_fn_96050�
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_96076�
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
(__inference_conv2d_9_layer_call_fn_96100�
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_96126�
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
5__inference_batch_normalization_1_layer_call_fn_96139
5__inference_batch_normalization_1_layer_call_fn_96152
5__inference_batch_normalization_1_layer_call_fn_96165
5__inference_batch_normalization_1_layer_call_fn_96178�
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
�2�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96196
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96214
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96232
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96250�
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
)__inference_flatten_1_layer_call_fn_96255�
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
D__inference_flatten_1_layer_call_and_return_conditional_losses_96261�
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
'__inference_dense_4_layer_call_fn_96270�
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
B__inference_dense_4_layer_call_and_return_conditional_losses_96281�
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
'__inference_dense_5_layer_call_fn_96290�
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
B__inference_dense_5_layer_call_and_return_conditional_losses_96301�
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
__inference_loss_fn_0_96321�
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
__inference_loss_fn_1_96341�
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
__inference_loss_fn_2_96361�
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
__inference_loss_fn_3_96381�
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
__inference_loss_fn_4_96401�
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
#__inference_signature_wrapper_95506input_3"�
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
 __inference__wrapped_model_94358�"#()/012;<AB8�5
.�+
)�&
input_3���������
� "2�/
-
dense_5"�
dense_5�����������
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96196�/012M�J
C�@
:�7
inputs+���������������������������H
p 
� "?�<
5�2
0+���������������������������H
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96214�/012M�J
C�@
:�7
inputs+���������������������������H
p
� "?�<
5�2
0+���������������������������H
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96232r/012;�8
1�.
(�%
inputs���������H
p 
� "-�*
#� 
0���������H
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_96250r/012;�8
1�.
(�%
inputs���������H
p
� "-�*
#� 
0���������H
� �
5__inference_batch_normalization_1_layer_call_fn_96139�/012M�J
C�@
:�7
inputs+���������������������������H
p 
� "2�/+���������������������������H�
5__inference_batch_normalization_1_layer_call_fn_96152�/012M�J
C�@
:�7
inputs+���������������������������H
p
� "2�/+���������������������������H�
5__inference_batch_normalization_1_layer_call_fn_96165e/012;�8
1�.
(�%
inputs���������H
p 
� " ����������H�
5__inference_batch_normalization_1_layer_call_fn_96178e/012;�8
1�.
(�%
inputs���������H
p
� " ����������H�
C__inference_conv2d_5_layer_call_and_return_conditional_losses_95926l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
(__inference_conv2d_5_layer_call_fn_95900_7�4
-�*
(�%
inputs���������
� " �����������
C__inference_conv2d_6_layer_call_and_return_conditional_losses_95976l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
(__inference_conv2d_6_layer_call_fn_95950_7�4
-�*
(�%
inputs���������
� " ���������� �
C__inference_conv2d_7_layer_call_and_return_conditional_losses_96026l7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������0
� �
(__inference_conv2d_7_layer_call_fn_96000_7�4
-�*
(�%
inputs��������� 
� " ����������0�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_96076l"#7�4
-�*
(�%
inputs���������0
� "-�*
#� 
0���������@
� �
(__inference_conv2d_8_layer_call_fn_96050_"#7�4
-�*
(�%
inputs���������0
� " ����������@�
C__inference_conv2d_9_layer_call_and_return_conditional_losses_96126l()7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������H
� �
(__inference_conv2d_9_layer_call_fn_96100_()7�4
-�*
(�%
inputs���������@
� " ����������H�
B__inference_dense_4_layer_call_and_return_conditional_losses_96281^;<0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dense_4_layer_call_fn_96270Q;<0�-
&�#
!�
inputs����������
� "������������
B__inference_dense_5_layer_call_and_return_conditional_losses_96301^AB0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dense_5_layer_call_fn_96290QAB0�-
&�#
!�
inputs����������
� "������������
B__inference_encoder_layer_call_and_return_conditional_losses_95264~"#()/012;<AB@�=
6�3
)�&
input_3���������
p 

 
� "&�#
�
0����������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_95388~"#()/012;<AB@�=
6�3
)�&
input_3���������
p

 
� "&�#
�
0����������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_95732}"#()/012;<AB?�<
5�2
(�%
inputs���������
p 

 
� "&�#
�
0����������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_95876}"#()/012;<AB?�<
5�2
(�%
inputs���������
p

 
� "&�#
�
0����������
� �
'__inference_encoder_layer_call_fn_94773q"#()/012;<AB@�=
6�3
)�&
input_3���������
p 

 
� "������������
'__inference_encoder_layer_call_fn_95140q"#()/012;<AB@�=
6�3
)�&
input_3���������
p

 
� "������������
'__inference_encoder_layer_call_fn_95547p"#()/012;<AB?�<
5�2
(�%
inputs���������
p 

 
� "������������
'__inference_encoder_layer_call_fn_95588p"#()/012;<AB?�<
5�2
(�%
inputs���������
p

 
� "������������
D__inference_flatten_1_layer_call_and_return_conditional_losses_96261a7�4
-�*
(�%
inputs���������H
� "&�#
�
0����������
� �
)__inference_flatten_1_layer_call_fn_96255T7�4
-�*
(�%
inputs���������H
� "�����������:
__inference_loss_fn_0_96321�

� 
� "� :
__inference_loss_fn_1_96341�

� 
� "� :
__inference_loss_fn_2_96361�

� 
� "� :
__inference_loss_fn_3_96381"�

� 
� "� :
__inference_loss_fn_4_96401(�

� 
� "� �
#__inference_signature_wrapper_95506�"#()/012;<ABC�@
� 
9�6
4
input_3)�&
input_3���������"2�/
-
dense_5"�
dense_5����������