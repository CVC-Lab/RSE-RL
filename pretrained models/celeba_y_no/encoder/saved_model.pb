щ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
2"
Utype:
2"
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
@
Softplus
features"T
activations"T"
Ttype:
2
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
 �"serve*2.9.22v2.9.1-132-g18960c44ad38��
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:H*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�H*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�H*
dtype0
d
z/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namez/bias
]
z/bias/Read/ReadVariableOpReadVariableOpz/bias*
_output_shapes
:H*
dtype0
m
z/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�H*
shared_name
z/kernel
f
z/kernel/Read/ReadVariableOpReadVariableOpz/kernel*
_output_shapes
:	�H*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
�	�*
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
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:H*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@H* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@H*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
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
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
�C
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�C
value�BB�B B�B
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
layer-6
layer_with_weights-5
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
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias*
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias*
�
0
1
"2
#3
+4
,5
46
57
>8
?9
@10
A11
N12
O13
V14
W15
^16
_17*
z
0
1
"2
#3
+4
,5
46
57
>8
?9
N10
O11
V12
W13
^14
_15*

`0
a1
b2
c3* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
itrace_0
jtrace_1
ktrace_2
ltrace_3* 
6
mtrace_0
ntrace_1
otrace_2
ptrace_3* 
* 

qserving_default* 

0
1*

0
1*
	
`0* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

"0
#1*

"0
#1*
	
a0* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

+0
,1*

+0
,1*
	
b0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

40
51*

40
51*
	
c0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
>0
?1
@2
A3*

>0
?1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

N0
O1*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
XR
VARIABLE_VALUEz/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEz/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

^0
_1*

^0
_1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

@0
A1*
J
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
9*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
`0* 
* 
* 
* 
* 
* 
* 
	
a0* 
* 
* 
* 
* 
* 
* 
	
b0* 
* 
* 
* 
* 
* 
* 
	
c0* 
* 
* 
* 

@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasz/kernelz/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������H:���������H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_213537
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpz/kernel/Read/ReadVariableOpz/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8� *(
f#R!
__inference__traced_save_214371
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasz/kernelz/biasdense_1/kerneldense_1/bias*
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
GPU2*0J 8� *+
f&R$
"__inference__traced_restore_214435��
�
�
(__inference_encoder_layer_call_fn_213212
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
�	�

unknown_12:	�

unknown_13:	�H

unknown_14:H

unknown_15:	�H

unknown_16:H
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������H:���������H*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_213128o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
__inference_loss_fn_0_214233O
5conv2d_kernel_regularizer_abs_readvariableop_resource:
identity��,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOpd
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5conv2d_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
�
�
(__inference_encoder_layer_call_fn_212897
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
�	�

unknown_12:	�

unknown_13:	�H

unknown_14:H

unknown_15:	�H

unknown_16:H
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������H:���������H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_212856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�}
�
C__inference_encoder_layer_call_and_return_conditional_losses_213128

inputs'
conv2d_213021:
conv2d_213023:)
conv2d_1_213026: 
conv2d_1_213028: )
conv2d_2_213031: @
conv2d_2_213033:@)
conv2d_3_213036:@H
conv2d_3_213038:H(
batch_normalization_213041:H(
batch_normalization_213043:H(
batch_normalization_213045:H(
batch_normalization_213047:H 
dense_213051:
�	�
dense_213053:	�!
dense_1_213056:	�H
dense_1_213058:H
z_213061:	�H
z_213063:H
identity

identity_1��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�z/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_213021conv2d_213023*
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
GPU2*0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_212625�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_213026conv2d_1_213028*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_212657�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_213031conv2d_2_213033*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_212689�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_213036conv2d_3_213038*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_212721�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_213041batch_normalization_213043batch_normalization_213045batch_normalization_213047*
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
GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212581�
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
GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_212742�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_213051dense_213053*
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
GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_212755�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_213056dense_1_213058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_212772�
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_213061z_213063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_212788d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_213021*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_213021*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_213026*&
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
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_213026*&
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
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_213031*&
_output_shapes
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_213031*&
_output_shapes
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_213036*&
_output_shapes
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_213036*&
_output_shapes
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hy

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������H�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_213640

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
�	�

unknown_12:	�

unknown_13:	�H

unknown_14:H

unknown_15:	�H

unknown_16:H
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������H:���������H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_212856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�}
�
C__inference_encoder_layer_call_and_return_conditional_losses_213322
input_1'
conv2d_213215:
conv2d_213217:)
conv2d_1_213220: 
conv2d_1_213222: )
conv2d_2_213225: @
conv2d_2_213227:@)
conv2d_3_213230:@H
conv2d_3_213232:H(
batch_normalization_213235:H(
batch_normalization_213237:H(
batch_normalization_213239:H(
batch_normalization_213241:H 
dense_213245:
�	�
dense_213247:	�!
dense_1_213250:	�H
dense_1_213252:H
z_213255:	�H
z_213257:H
identity

identity_1��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�z/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_213215conv2d_213217*
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
GPU2*0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_212625�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_213220conv2d_1_213222*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_212657�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_213225conv2d_2_213227*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_212689�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_213230conv2d_3_213232*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_212721�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_213235batch_normalization_213237batch_normalization_213239batch_normalization_213241*
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
GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212550�
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
GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_212742�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_213245dense_213247*
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
GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_212755�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_213250dense_1_213252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_212772�
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_213255z_213257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_212788d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_213215*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_213215*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_213220*&
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
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_213220*&
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
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_213225*&
_output_shapes
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_213225*&
_output_shapes
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_213230*&
_output_shapes
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_213230*&
_output_shapes
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hy

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������H�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
B__inference_conv2d_layer_call_and_return_conditional_losses_212625

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_213537
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
�	�

unknown_12:	�

unknown_13:	�H

unknown_14:H

unknown_15:	�H

unknown_16:H
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������H:���������H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_212528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
4__inference_batch_normalization_layer_call_fn_214094

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
GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212550�
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
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214143

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
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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
=__inference_z_layer_call_and_return_conditional_losses_212788

inputs1
matmul_readvariableop_resource:	�H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214125

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
4__inference_batch_normalization_layer_call_fn_214107

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
GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212581�
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
�
D
(__inference_flatten_layer_call_fn_214148

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
GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_212742a
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
 
_user_specified_nameinputs
�}
�
C__inference_encoder_layer_call_and_return_conditional_losses_213432
input_1'
conv2d_213325:
conv2d_213327:)
conv2d_1_213330: 
conv2d_1_213332: )
conv2d_2_213335: @
conv2d_2_213337:@)
conv2d_3_213340:@H
conv2d_3_213342:H(
batch_normalization_213345:H(
batch_normalization_213347:H(
batch_normalization_213349:H(
batch_normalization_213351:H 
dense_213355:
�	�
dense_213357:	�!
dense_1_213360:	�H
dense_1_213362:H
z_213365:	�H
z_213367:H
identity

identity_1��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�z/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_213325conv2d_213327*
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
GPU2*0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_212625�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_213330conv2d_1_213332*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_212657�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_213335conv2d_2_213337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_212689�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_213340conv2d_3_213342*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_212721�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_213345batch_normalization_213347batch_normalization_213349batch_normalization_213351*
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
GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212581�
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
GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_212742�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_213355dense_213357*
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
GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_212755�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_213360dense_1_213362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_212772�
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_213365z_213367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_212788d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_213325*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_213325*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_213330*&
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
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_213330*&
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
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_213335*&
_output_shapes
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_213335*&
_output_shapes
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_213340*&
_output_shapes
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_213340*&
_output_shapes
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hy

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������H�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
)__inference_conv2d_3_layer_call_fn_214055

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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_212721w
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
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_conv2d_1_layer_call_fn_213985

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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_212657w
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
�
�
&__inference_dense_layer_call_fn_214163

inputs
unknown:
�	�
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
GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_212755p
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
:����������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_212742

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
"__inference_z_layer_call_fn_214183

inputs
unknown:	�H
	unknown_0:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_212788o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_212755

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
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
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_212772

inputs1
matmul_readvariableop_resource:	�H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������HX
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������He
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_214154

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
�\
�
!__inference__wrapped_model_212528
input_1G
-encoder_conv2d_conv2d_readvariableop_resource:<
.encoder_conv2d_biasadd_readvariableop_resource:I
/encoder_conv2d_1_conv2d_readvariableop_resource: >
0encoder_conv2d_1_biasadd_readvariableop_resource: I
/encoder_conv2d_2_conv2d_readvariableop_resource: @>
0encoder_conv2d_2_biasadd_readvariableop_resource:@I
/encoder_conv2d_3_conv2d_readvariableop_resource:@H>
0encoder_conv2d_3_biasadd_readvariableop_resource:HA
3encoder_batch_normalization_readvariableop_resource:HC
5encoder_batch_normalization_readvariableop_1_resource:HR
Dencoder_batch_normalization_fusedbatchnormv3_readvariableop_resource:HT
Fencoder_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H@
,encoder_dense_matmul_readvariableop_resource:
�	�<
-encoder_dense_biasadd_readvariableop_resource:	�A
.encoder_dense_1_matmul_readvariableop_resource:	�H=
/encoder_dense_1_biasadd_readvariableop_resource:H;
(encoder_z_matmul_readvariableop_resource:	�H7
)encoder_z_biasadd_readvariableop_resource:H
identity

identity_1��;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp�=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�*encoder/batch_normalization/ReadVariableOp�,encoder/batch_normalization/ReadVariableOp_1�%encoder/conv2d/BiasAdd/ReadVariableOp�$encoder/conv2d/Conv2D/ReadVariableOp�'encoder/conv2d_1/BiasAdd/ReadVariableOp�&encoder/conv2d_1/Conv2D/ReadVariableOp�'encoder/conv2d_2/BiasAdd/ReadVariableOp�&encoder/conv2d_2/Conv2D/ReadVariableOp�'encoder/conv2d_3/BiasAdd/ReadVariableOp�&encoder/conv2d_3/Conv2D/ReadVariableOp�$encoder/dense/BiasAdd/ReadVariableOp�#encoder/dense/MatMul/ReadVariableOp�&encoder/dense_1/BiasAdd/ReadVariableOp�%encoder/dense_1/MatMul/ReadVariableOp� encoder/z/BiasAdd/ReadVariableOp�encoder/z/MatMul/ReadVariableOp�
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
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
: @*
dtype0�
encoder/conv2d_2/Conv2DConv2D#encoder/conv2d_1/Relu:activations:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@z
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
encoder/conv2d_3/Conv2DConv2D#encoder/conv2d_2/Relu:activations:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hz
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
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
,encoder/batch_normalization/FusedBatchNormV3FusedBatchNormV3#encoder/conv2d_3/Relu:activations:02encoder/batch_normalization/ReadVariableOp:value:04encoder/batch_normalization/ReadVariableOp_1:value:0Cencoder/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Eencoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
�	�*
dtype0�
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0�
encoder/dense_1/MatMulMatMul encoder/dense/Relu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H�
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hx
encoder/dense_1/SoftplusSoftplus encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������H�
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0�
encoder/z/MatMulMatMul encoder/dense/Relu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H�
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hu
IdentityIdentity&encoder/dense_1/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������Hk

Identity_1Identityencoder/z/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������H�
NoOpNoOp<^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp>^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^encoder/batch_normalization/ReadVariableOp-^encoder/batch_normalization/ReadVariableOp_1&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp!^encoder/z/BiasAdd/ReadVariableOp ^encoder/z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2z
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
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp2D
 encoder/z/BiasAdd/ReadVariableOp encoder/z/BiasAdd/ReadVariableOp2B
encoder/z/MatMul/ReadVariableOpencoder/z/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_214011

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

�
C__inference_dense_1_layer_call_and_return_conditional_losses_214213

inputs1
matmul_readvariableop_resource:	�H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������HX
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������He
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_214273Q
7conv2d_2_kernel_regularizer_abs_readvariableop_resource: @
identity��.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
�-
�
__inference__traced_save_214371
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop'
#savev2_z_kernel_read_readvariableop%
!savev2_z_bias_read_readvariableop-
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
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_z_kernel_read_readvariableop!savev2_z_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�: ::: : : @:@:@H:H:H:H:H:H:
�	�:�:	�H:H:	�H:H: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 
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
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@H: 

_output_shapes
:H: 	

_output_shapes
:H: 


_output_shapes
:H: 

_output_shapes
:H: 

_output_shapes
:H:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:%!

_output_shapes
:	�H: 

_output_shapes
:H:%!

_output_shapes
:	�H: 

_output_shapes
:H:

_output_shapes
: 
�

�
A__inference_dense_layer_call_and_return_conditional_losses_214174

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
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
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_213683

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
�	�

unknown_12:	�

unknown_13:	�H

unknown_14:H

unknown_15:	�H

unknown_16:H
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������H:���������H*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_213128o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_212657

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
�}
�
C__inference_encoder_layer_call_and_return_conditional_losses_212856

inputs'
conv2d_212626:
conv2d_212628:)
conv2d_1_212658: 
conv2d_1_212660: )
conv2d_2_212690: @
conv2d_2_212692:@)
conv2d_3_212722:@H
conv2d_3_212724:H(
batch_normalization_212727:H(
batch_normalization_212729:H(
batch_normalization_212731:H(
batch_normalization_212733:H 
dense_212756:
�	�
dense_212758:	�!
dense_1_212773:	�H
dense_1_212775:H
z_212789:	�H
z_212791:H
identity

identity_1��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp� conv2d_1/StatefulPartitionedCall�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp� conv2d_2/StatefulPartitionedCall�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp� conv2d_3/StatefulPartitionedCall�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�z/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_212626conv2d_212628*
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
GPU2*0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_212625�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_212658conv2d_1_212660*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_212657�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_212690conv2d_2_212692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_212689�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_212722conv2d_3_212724*
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
GPU2*0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_212721�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_212727batch_normalization_212729batch_normalization_212731batch_normalization_212733*
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
GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212550�
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
GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_212742�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_212756dense_212758*
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
GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_212755�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_212773dense_1_212775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_212772�
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_212789z_212791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_212788d
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_212626*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_212626*&
_output_shapes
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_212658*&
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
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_212658*&
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
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_212690*&
_output_shapes
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_212690*&
_output_shapes
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_212722*&
_output_shapes
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_212722*&
_output_shapes
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������Hy

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������H�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2Z
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_212689

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
:���������@�
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
�
�
'__inference_conv2d_layer_call_fn_213950

inputs!
unknown:
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
GPU2*0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_212625w
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
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
=__inference_z_layer_call_and_return_conditional_losses_214193

inputs1
matmul_readvariableop_resource:	�H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_212721

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp|
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
r
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
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
:���������H�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212581

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
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
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
Ц
�
C__inference_encoder_layer_call_and_return_conditional_losses_213941

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@H6
(conv2d_3_biasadd_readvariableop_resource:H9
+batch_normalization_readvariableop_resource:H;
-batch_normalization_readvariableop_1_resource:HJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:HL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H8
$dense_matmul_readvariableop_resource:
�	�4
%dense_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�H5
'dense_1_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	�H/
!z_biasadd_readvariableop_resource:H
identity

identity_1��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�z/BiasAdd/ReadVariableOp�z/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
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
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hj
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
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
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(^
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
�	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hh
dense_1/SoftplusSoftplusdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������Hy
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0
z/MatMulMatMuldense/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hv
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hd
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
: a
IdentityIdentityz/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Ho

Identity_1Identitydense_1/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������H�	
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2H
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_212550

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
�
B__inference_conv2d_layer_call_and_return_conditional_losses_213976

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_214293Q
7conv2d_3_kernel_regularizer_abs_readvariableop_resource:@H
identity��.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
�I
�
"__inference__traced_restore_214435
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel: .
 assignvariableop_3_conv2d_1_bias: <
"assignvariableop_4_conv2d_2_kernel: @.
 assignvariableop_5_conv2d_2_bias:@<
"assignvariableop_6_conv2d_3_kernel:@H.
 assignvariableop_7_conv2d_3_bias:H:
,assignvariableop_8_batch_normalization_gamma:H9
+assignvariableop_9_batch_normalization_beta:HA
3assignvariableop_10_batch_normalization_moving_mean:HE
7assignvariableop_11_batch_normalization_moving_variance:H4
 assignvariableop_12_dense_kernel:
�	�-
assignvariableop_13_dense_bias:	�/
assignvariableop_14_z_kernel:	�H(
assignvariableop_15_z_bias:H5
"assignvariableop_16_dense_1_kernel:	�H.
 assignvariableop_17_dense_1_bias:H
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
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
AssignVariableOp_8AssignVariableOp,assignvariableop_8_batch_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_batch_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp3assignvariableop_10_batch_normalization_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp7assignvariableop_11_batch_normalization_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_z_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_z_biasIdentity_15:output:0"/device:CPU:0*
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
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_214081

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp|
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
r
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
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
:���������H�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_214253Q
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
�
�
(__inference_dense_1_layer_call_fn_214202

inputs
unknown:	�H
	unknown_0:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_212772o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
֟
�
C__inference_encoder_layer_call_and_return_conditional_losses_213812

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@H6
(conv2d_3_biasadd_readvariableop_resource:H9
+batch_normalization_readvariableop_resource:H;
-batch_normalization_readvariableop_1_resource:HJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:HL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H8
$dense_matmul_readvariableop_resource:
�	�4
%dense_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�H5
'dense_1_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	�H/
!z_biasadd_readvariableop_resource:H
identity

identity_1��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�,conv2d/kernel/Regularizer/Abs/ReadVariableOp�/conv2d/kernel/Regularizer/Square/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_1/kernel/Regularizer/Square/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_3/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�z/BiasAdd/ReadVariableOp�z/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
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
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Hj
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
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
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
�	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hh
dense_1/SoftplusSoftplusdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������Hy
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	�H*
dtype0
z/MatMulMatMuldense/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hv
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Hd
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
:*
dtype0�
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:z
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
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
:@H*
dtype0�
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
:@H*
dtype0�
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H|
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
: a
IdentityIdentityz/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Ho

Identity_1Identitydense_1/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:���������H�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������: : : : : : : : : : : : : : : : : : 2j
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
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_2_layer_call_fn_214020

inputs!
unknown: @
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
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_212689w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
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
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_214046

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp�1conv2d_2/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@f
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
: @*
dtype0�
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @|
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
:���������@�
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
 
_user_specified_nameinputs"�L
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
serving_default_input_1:0���������;
dense_10
StatefulPartitionedCall:0���������H5
z0
StatefulPartitionedCall:1���������Htensorflow/serving/predict:��
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
layer-6
layer_with_weights-5
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
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias"
_tf_keras_layer
�
0
1
"2
#3
+4
,5
46
57
>8
?9
@10
A11
N12
O13
V14
W15
^16
_17"
trackable_list_wrapper
�
0
1
"2
#3
+4
,5
46
57
>8
?9
N10
O11
V12
W13
^14
_15"
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
itrace_0
jtrace_1
ktrace_2
ltrace_32�
(__inference_encoder_layer_call_fn_212897
(__inference_encoder_layer_call_fn_213640
(__inference_encoder_layer_call_fn_213683
(__inference_encoder_layer_call_fn_213212�
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
 zitrace_0zjtrace_1zktrace_2zltrace_3
�
mtrace_0
ntrace_1
otrace_2
ptrace_32�
C__inference_encoder_layer_call_and_return_conditional_losses_213812
C__inference_encoder_layer_call_and_return_conditional_losses_213941
C__inference_encoder_layer_call_and_return_conditional_losses_213322
C__inference_encoder_layer_call_and_return_conditional_losses_213432�
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
 zmtrace_0zntrace_1zotrace_2zptrace_3
�B�
!__inference__wrapped_model_212528input_1"�
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
,
qserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
wtrace_02�
'__inference_conv2d_layer_call_fn_213950�
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
 zwtrace_0
�
xtrace_02�
B__inference_conv2d_layer_call_and_return_conditional_losses_213976�
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
 zxtrace_0
':%2conv2d/kernel
:2conv2d/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
'
a0"
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
)__inference_conv2d_1_layer_call_fn_213985�
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
 z~trace_0
�
trace_02�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_214011�
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
 ztrace_0
):' 2conv2d_1/kernel
: 2conv2d_1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
'
b0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_2_layer_call_fn_214020�
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
 z�trace_0
�
�trace_02�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_214046�
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
 z�trace_0
):' @2conv2d_2/kernel
:@2conv2d_2/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_3_layer_call_fn_214055�
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
 z�trace_0
�
�trace_02�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_214081�
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
 z�trace_0
):'@H2conv2d_3/kernel
:H2conv2d_3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_214094
4__inference_batch_normalization_layer_call_fn_214107�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214125
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214143�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
':%H2batch_normalization/gamma
&:$H2batch_normalization/beta
/:-H (2batch_normalization/moving_mean
3:1H (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_214148�
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
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_214154�
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
 z�trace_0
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_214163�
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
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_214174�
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
 z�trace_0
 :
�	�2dense/kernel
:�2
dense/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_z_layer_call_fn_214183�
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
 z�trace_0
�
�trace_02�
=__inference_z_layer_call_and_return_conditional_losses_214193�
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
 z�trace_0
:	�H2z/kernel
:H2z/bias
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_214202�
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
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_214213�
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
 z�trace_0
!:	�H2dense_1/kernel
:H2dense_1/bias
�
�trace_02�
__inference_loss_fn_0_214233�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_214253�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_214273�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_214293�
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
annotations� *� z�trace_0
.
@0
A1"
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
�B�
(__inference_encoder_layer_call_fn_212897input_1"�
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
(__inference_encoder_layer_call_fn_213640inputs"�
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
(__inference_encoder_layer_call_fn_213683inputs"�
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
(__inference_encoder_layer_call_fn_213212input_1"�
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
�B�
C__inference_encoder_layer_call_and_return_conditional_losses_213812inputs"�
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
�B�
C__inference_encoder_layer_call_and_return_conditional_losses_213941inputs"�
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
�B�
C__inference_encoder_layer_call_and_return_conditional_losses_213322input_1"�
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
�B�
C__inference_encoder_layer_call_and_return_conditional_losses_213432input_1"�
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
$__inference_signature_wrapper_213537input_1"�
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2d_layer_call_fn_213950inputs"�
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
�B�
B__inference_conv2d_layer_call_and_return_conditional_losses_213976inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
a0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_1_layer_call_fn_213985inputs"�
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
�B�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_214011inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
b0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_2_layer_call_fn_214020inputs"�
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
�B�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_214046inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_3_layer_call_fn_214055inputs"�
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
�B�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_214081inputs"�
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
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_layer_call_fn_214094inputs"�
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
�B�
4__inference_batch_normalization_layer_call_fn_214107inputs"�
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
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214125inputs"�
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
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214143inputs"�
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
�B�
(__inference_flatten_layer_call_fn_214148inputs"�
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
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_214154inputs"�
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
�B�
&__inference_dense_layer_call_fn_214163inputs"�
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
�B�
A__inference_dense_layer_call_and_return_conditional_losses_214174inputs"�
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
�B�
"__inference_z_layer_call_fn_214183inputs"�
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
�B�
=__inference_z_layer_call_and_return_conditional_losses_214193inputs"�
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
�B�
(__inference_dense_1_layer_call_fn_214202inputs"�
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
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_214213inputs"�
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
�B�
__inference_loss_fn_0_214233"�
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
__inference_loss_fn_1_214253"�
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
__inference_loss_fn_2_214273"�
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
__inference_loss_fn_3_214293"�
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
annotations� *� �
!__inference__wrapped_model_212528�"#+,45>?@ANO^_VW8�5
.�+
)�&
input_1���������
� "S�P
,
dense_1!�
dense_1���������H
 
z�
z���������H�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214125�>?@AM�J
C�@
:�7
inputs+���������������������������H
p 
� "?�<
5�2
0+���������������������������H
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_214143�>?@AM�J
C�@
:�7
inputs+���������������������������H
p
� "?�<
5�2
0+���������������������������H
� �
4__inference_batch_normalization_layer_call_fn_214094�>?@AM�J
C�@
:�7
inputs+���������������������������H
p 
� "2�/+���������������������������H�
4__inference_batch_normalization_layer_call_fn_214107�>?@AM�J
C�@
:�7
inputs+���������������������������H
p
� "2�/+���������������������������H�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_214011l"#7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
)__inference_conv2d_1_layer_call_fn_213985_"#7�4
-�*
(�%
inputs���������
� " ���������� �
D__inference_conv2d_2_layer_call_and_return_conditional_losses_214046l+,7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
)__inference_conv2d_2_layer_call_fn_214020_+,7�4
-�*
(�%
inputs��������� 
� " ����������@�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_214081l457�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������H
� �
)__inference_conv2d_3_layer_call_fn_214055_457�4
-�*
(�%
inputs���������@
� " ����������H�
B__inference_conv2d_layer_call_and_return_conditional_losses_213976l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
'__inference_conv2d_layer_call_fn_213950_7�4
-�*
(�%
inputs���������
� " �����������
C__inference_dense_1_layer_call_and_return_conditional_losses_214213]^_0�-
&�#
!�
inputs����������
� "%�"
�
0���������H
� |
(__inference_dense_1_layer_call_fn_214202P^_0�-
&�#
!�
inputs����������
� "����������H�
A__inference_dense_layer_call_and_return_conditional_losses_214174^NO0�-
&�#
!�
inputs����������	
� "&�#
�
0����������
� {
&__inference_dense_layer_call_fn_214163QNO0�-
&�#
!�
inputs����������	
� "������������
C__inference_encoder_layer_call_and_return_conditional_losses_213322�"#+,45>?@ANO^_VW@�=
6�3
)�&
input_1���������
p 

 
� "K�H
A�>
�
0/0���������H
�
0/1���������H
� �
C__inference_encoder_layer_call_and_return_conditional_losses_213432�"#+,45>?@ANO^_VW@�=
6�3
)�&
input_1���������
p

 
� "K�H
A�>
�
0/0���������H
�
0/1���������H
� �
C__inference_encoder_layer_call_and_return_conditional_losses_213812�"#+,45>?@ANO^_VW?�<
5�2
(�%
inputs���������
p 

 
� "K�H
A�>
�
0/0���������H
�
0/1���������H
� �
C__inference_encoder_layer_call_and_return_conditional_losses_213941�"#+,45>?@ANO^_VW?�<
5�2
(�%
inputs���������
p

 
� "K�H
A�>
�
0/0���������H
�
0/1���������H
� �
(__inference_encoder_layer_call_fn_212897�"#+,45>?@ANO^_VW@�=
6�3
)�&
input_1���������
p 

 
� "=�:
�
0���������H
�
1���������H�
(__inference_encoder_layer_call_fn_213212�"#+,45>?@ANO^_VW@�=
6�3
)�&
input_1���������
p

 
� "=�:
�
0���������H
�
1���������H�
(__inference_encoder_layer_call_fn_213640�"#+,45>?@ANO^_VW?�<
5�2
(�%
inputs���������
p 

 
� "=�:
�
0���������H
�
1���������H�
(__inference_encoder_layer_call_fn_213683�"#+,45>?@ANO^_VW?�<
5�2
(�%
inputs���������
p

 
� "=�:
�
0���������H
�
1���������H�
C__inference_flatten_layer_call_and_return_conditional_losses_214154a7�4
-�*
(�%
inputs���������H
� "&�#
�
0����������	
� �
(__inference_flatten_layer_call_fn_214148T7�4
-�*
(�%
inputs���������H
� "�����������	;
__inference_loss_fn_0_214233�

� 
� "� ;
__inference_loss_fn_1_214253"�

� 
� "� ;
__inference_loss_fn_2_214273+�

� 
� "� ;
__inference_loss_fn_3_2142934�

� 
� "� �
$__inference_signature_wrapper_213537�"#+,45>?@ANO^_VWC�@
� 
9�6
4
input_1)�&
input_1���������"S�P
,
dense_1!�
dense_1���������H
 
z�
z���������H�
=__inference_z_layer_call_and_return_conditional_losses_214193]VW0�-
&�#
!�
inputs����������
� "%�"
�
0���������H
� v
"__inference_z_layer_call_fn_214183PVW0�-
&�#
!�
inputs����������
� "����������H