��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
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
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
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
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628ɋ
�
ai_model_6711/dense_13526/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name ai_model_6711/dense_13526/bias
�
2ai_model_6711/dense_13526/bias/Read/ReadVariableOpReadVariableOpai_model_6711/dense_13526/bias*
_output_shapes
:*
dtype0
�
 ai_model_6711/dense_13526/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" ai_model_6711/dense_13526/kernel
�
4ai_model_6711/dense_13526/kernel/Read/ReadVariableOpReadVariableOp ai_model_6711/dense_13526/kernel*
_output_shapes

:@*
dtype0
�
2ai_model_6711/simple_rnn_6710/simple_rnn_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias
�
Fai_model_6711/simple_rnn_6710/simple_rnn_cell/bias/Read/ReadVariableOpReadVariableOp2ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias*
_output_shapes
:@*
dtype0
�
>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*O
shared_name@>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel
�
Rai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel*
_output_shapes

:@@*
dtype0
�
4ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*E
shared_name64ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel
�
Hai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel/Read/ReadVariableOpReadVariableOp4ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel*
_output_shapes

:@*
dtype0
�
ai_model_6711/dense_13525/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name ai_model_6711/dense_13525/bias
�
2ai_model_6711/dense_13525/bias/Read/ReadVariableOpReadVariableOpai_model_6711/dense_13525/bias*
_output_shapes
:*
dtype0
�
 ai_model_6711/dense_13525/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" ai_model_6711/dense_13525/kernel
�
4ai_model_6711/dense_13525/kernel/Read/ReadVariableOpReadVariableOp ai_model_6711/dense_13525/kernel*
_output_shapes

:*
dtype0
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1 ai_model_6711/dense_13525/kernelai_model_6711/dense_13525/bias4ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel2ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel ai_model_6711/dense_13526/kernelai_model_6711/dense_13526/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_75823718

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1
	rnn


dense2

signatures*
5
0
1
2
3
4
5
6*
5
0
1
2
3
4
5
6*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

kernel
bias*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(cell
)
state_spec*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias*

0serving_default* 
`Z
VARIABLE_VALUE ai_model_6711/dense_13525/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEai_model_6711/dense_13525/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE2ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE ai_model_6711/dense_13526/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEai_model_6711/dense_13526/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

6trace_0* 

7trace_0* 

0
1
2*

0
1
2*
* 
�

8states
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
6
Btrace_0
Ctrace_1
Dtrace_2
Etrace_3* 
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_random_generator

kernel
recurrent_kernel
bias*
* 

0
1*

0
1*
* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
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

(0*
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

0
1
2*

0
1
2*
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

Ytrace_0
Ztrace_1* 

[trace_0
\trace_1* 
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename ai_model_6711/dense_13525/kernelai_model_6711/dense_13525/bias4ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel2ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias ai_model_6711/dense_13526/kernelai_model_6711/dense_13526/biasConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_75824400
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename ai_model_6711/dense_13525/kernelai_model_6711/dense_13525/bias4ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel2ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias ai_model_6711/dense_13526/kernelai_model_6711/dense_13526/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_75824430��
�
�
while_cond_75823843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75823843___redundant_placeholder06
2while_while_cond_75823843___redundant_placeholder16
2while_while_cond_75823843___redundant_placeholder26
2while_while_cond_75823843___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
.__inference_dense_13525_layer_call_fn_75823727

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823333s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823723:($
"
_user_specified_name
75823721:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_simple_rnn_cell_layer_call_fn_75824288

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823055o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75824282:($
"
_user_specified_name
75824280:($
"
_user_specified_name
75824278:QM
'
_output_shapes
:���������@
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_dense_13526_layer_call_fn_75824243

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_13526_layer_call_and_return_conditional_losses_75823484s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75824239:($
"
_user_specified_name
75824237:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_75824059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75824059___redundant_placeholder06
2while_while_cond_75824059___redundant_placeholder16
2while_while_cond_75824059___redundant_placeholder26
2while_while_cond_75824059___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�)
�
$__inference__traced_restore_75824430
file_prefixC
1assignvariableop_ai_model_6711_dense_13525_kernel:?
1assignvariableop_1_ai_model_6711_dense_13525_bias:Y
Gassignvariableop_2_ai_model_6711_simple_rnn_6710_simple_rnn_cell_kernel:@c
Qassignvariableop_3_ai_model_6711_simple_rnn_6710_simple_rnn_cell_recurrent_kernel:@@S
Eassignvariableop_4_ai_model_6711_simple_rnn_6710_simple_rnn_cell_bias:@E
3assignvariableop_5_ai_model_6711_dense_13526_kernel:@?
1assignvariableop_6_ai_model_6711_dense_13526_bias:

identity_8��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp1assignvariableop_ai_model_6711_dense_13525_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp1assignvariableop_1_ai_model_6711_dense_13525_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpGassignvariableop_2_ai_model_6711_simple_rnn_6710_simple_rnn_cell_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpQassignvariableop_3_ai_model_6711_simple_rnn_6710_simple_rnn_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpEassignvariableop_4_ai_model_6711_simple_rnn_6710_simple_rnn_cell_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp3assignvariableop_5_ai_model_6711_dense_13526_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp1assignvariableop_6_ai_model_6711_dense_13526_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*
_output_shapes
 "!

identity_8Identity_8:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62$
AssignVariableOpAssignVariableOp:>:
8
_user_specified_name ai_model_6711/dense_13526/bias:@<
:
_user_specified_name" ai_model_6711/dense_13526/kernel:RN
L
_user_specified_name42ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias:^Z
X
_user_specified_name@>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel:TP
N
_user_specified_name64ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel:>:
8
_user_specified_name ai_model_6711/dense_13525/bias:@<
:
_user_specified_name" ai_model_6711/dense_13525/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
2__inference_simple_rnn_6710_layer_call_fn_75823802

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823606s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823798:($
"
_user_specified_name
75823796:($
"
_user_specified_name
75823794:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_75823539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75823539___redundant_placeholder06
2while_while_cond_75823539___redundant_placeholder16
2while_while_cond_75823539___redundant_placeholder26
2while_while_cond_75823539___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
1ai_model_6711_simple_rnn_6710_while_cond_75822918X
Tai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_loop_counter^
Zai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_maximum_iterations3
/ai_model_6711_simple_rnn_6710_while_placeholder5
1ai_model_6711_simple_rnn_6710_while_placeholder_15
1ai_model_6711_simple_rnn_6710_while_placeholder_2Z
Vai_model_6711_simple_rnn_6710_while_less_ai_model_6711_simple_rnn_6710_strided_slice_1r
nai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_cond_75822918___redundant_placeholder0r
nai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_cond_75822918___redundant_placeholder1r
nai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_cond_75822918___redundant_placeholder2r
nai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_cond_75822918___redundant_placeholder30
,ai_model_6711_simple_rnn_6710_while_identity
�
(ai_model_6711/simple_rnn_6710/while/LessLess/ai_model_6711_simple_rnn_6710_while_placeholderVai_model_6711_simple_rnn_6710_while_less_ai_model_6711_simple_rnn_6710_strided_slice_1*
T0*
_output_shapes
: �
,ai_model_6711/simple_rnn_6710/while/IdentityIdentity,ai_model_6711/simple_rnn_6710/while/Less:z:0*
T0
*
_output_shapes
: "e
,ai_model_6711_simple_rnn_6710_while_identity5ai_model_6711/simple_rnn_6710/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::ea

_output_shapes
: 
G
_user_specified_name/-ai_model_6711/simple_rnn_6710/strided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :nj

_output_shapes
: 
P
_user_specified_name86ai_model_6711/simple_rnn_6710/while/maximum_iterations:h d

_output_shapes
: 
J
_user_specified_name20ai_model_6711/simple_rnn_6710/while/loop_counter
�
�
&__inference_signature_wrapper_75823718
input_1
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_75823012s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823714:($
"
_user_specified_name
75823712:($
"
_user_specified_name
75823710:($
"
_user_specified_name
75823708:($
"
_user_specified_name
75823706:($
"
_user_specified_name
75823704:($
"
_user_specified_name
75823702:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
2__inference_simple_rnn_6710_layer_call_fn_75823791

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823446s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823787:($
"
_user_specified_name
75823785:($
"
_user_specified_name
75823783:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823131

inputs*
simple_rnn_cell_75823056:@&
simple_rnn_cell_75823058:@*
simple_rnn_cell_75823060:@@
identity��'simple_rnn_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_75823056simple_rnn_cell_75823058simple_rnn_cell_75823060*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823055n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_75823056simple_rnn_cell_75823058simple_rnn_cell_75823060*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75823068*
condR
while_cond_75823067*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@T
NoOpNoOp(^simple_rnn_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2R
'simple_rnn_cell/StatefulPartitionedCall'simple_rnn_cell/StatefulPartitionedCall2
whilewhile:($
"
_user_specified_name
75823060:($
"
_user_specified_name
75823058:($
"
_user_specified_name
75823056:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�.
�
while_body_75824060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
6while_simple_rnn_cell_matmul_readvariableop_resource_0:@E
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:@J
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
4while_simple_rnn_cell_matmul_readvariableop_resource:@C
5while_simple_rnn_cell_biasadd_readvariableop_resource:@H
6while_simple_rnn_cell_matmul_1_readvariableop_resource:@@��,while/simple_rnn_cell/BiasAdd/ReadVariableOp�+while/simple_rnn_cell/MatMul/ReadVariableOp�-while/simple_rnn_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@s
while/simple_rnn_cell/ReluReluwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder(while/simple_rnn_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity(while/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823055

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
ReluReluadd:z:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823333

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
while_body_75823187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_75823209_0:@.
 while_simple_rnn_cell_75823211_0:@2
 while_simple_rnn_cell_75823213_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_75823209:@,
while_simple_rnn_cell_75823211:@0
while_simple_rnn_cell_75823213:@@��-while/simple_rnn_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_75823209_0 while_simple_rnn_cell_75823211_0 while_simple_rnn_cell_75823213_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823174�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder6while/simple_rnn_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity6while/simple_rnn_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@X

while/NoOpNoOp.^while/simple_rnn_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"B
while_simple_rnn_cell_75823209 while_simple_rnn_cell_75823209_0"B
while_simple_rnn_cell_75823211 while_simple_rnn_cell_75823211_0"B
while_simple_rnn_cell_75823213 while_simple_rnn_cell_75823213_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2^
-while/simple_rnn_cell/StatefulPartitionedCall-while/simple_rnn_cell/StatefulPartitionedCall:(	$
"
_user_specified_name
75823213:($
"
_user_specified_name
75823211:($
"
_user_specified_name
75823209:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
while_cond_75823379
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75823379___redundant_placeholder06
2while_while_cond_75823379___redundant_placeholder16
2while_while_cond_75823379___redundant_placeholder26
2while_while_cond_75823379___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823174

inputs

states0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
ReluReluadd:z:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824018
inputs_0@
.simple_rnn_cell_matmul_readvariableop_resource:@=
/simple_rnn_cell_biasadd_readvariableop_resource:@B
0simple_rnn_cell_matmul_1_readvariableop_resource:@@
identity��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75823952*
condR
while_cond_75823951*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@�
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
��
�
#__inference__wrapped_model_75823012
input_1M
;ai_model_6711_dense_13525_tensordot_readvariableop_resource:G
9ai_model_6711_dense_13525_biasadd_readvariableop_resource:^
Lai_model_6711_simple_rnn_6710_simple_rnn_cell_matmul_readvariableop_resource:@[
Mai_model_6711_simple_rnn_6710_simple_rnn_cell_biasadd_readvariableop_resource:@`
Nai_model_6711_simple_rnn_6710_simple_rnn_cell_matmul_1_readvariableop_resource:@@M
;ai_model_6711_dense_13526_tensordot_readvariableop_resource:@G
9ai_model_6711_dense_13526_biasadd_readvariableop_resource:
identity��0ai_model_6711/dense_13525/BiasAdd/ReadVariableOp�2ai_model_6711/dense_13525/Tensordot/ReadVariableOp�0ai_model_6711/dense_13526/BiasAdd/ReadVariableOp�2ai_model_6711/dense_13526/Tensordot/ReadVariableOp�Dai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAdd/ReadVariableOp�Cai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul/ReadVariableOp�Eai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1/ReadVariableOp�#ai_model_6711/simple_rnn_6710/while�
2ai_model_6711/dense_13525/Tensordot/ReadVariableOpReadVariableOp;ai_model_6711_dense_13525_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0r
(ai_model_6711/dense_13525/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(ai_model_6711/dense_13525/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
)ai_model_6711/dense_13525/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
::��s
1ai_model_6711/dense_13525/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,ai_model_6711/dense_13525/Tensordot/GatherV2GatherV22ai_model_6711/dense_13525/Tensordot/Shape:output:01ai_model_6711/dense_13525/Tensordot/free:output:0:ai_model_6711/dense_13525/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3ai_model_6711/dense_13525/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.ai_model_6711/dense_13525/Tensordot/GatherV2_1GatherV22ai_model_6711/dense_13525/Tensordot/Shape:output:01ai_model_6711/dense_13525/Tensordot/axes:output:0<ai_model_6711/dense_13525/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)ai_model_6711/dense_13525/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(ai_model_6711/dense_13525/Tensordot/ProdProd5ai_model_6711/dense_13525/Tensordot/GatherV2:output:02ai_model_6711/dense_13525/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+ai_model_6711/dense_13525/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*ai_model_6711/dense_13525/Tensordot/Prod_1Prod7ai_model_6711/dense_13525/Tensordot/GatherV2_1:output:04ai_model_6711/dense_13525/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/ai_model_6711/dense_13525/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*ai_model_6711/dense_13525/Tensordot/concatConcatV21ai_model_6711/dense_13525/Tensordot/free:output:01ai_model_6711/dense_13525/Tensordot/axes:output:08ai_model_6711/dense_13525/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)ai_model_6711/dense_13525/Tensordot/stackPack1ai_model_6711/dense_13525/Tensordot/Prod:output:03ai_model_6711/dense_13525/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-ai_model_6711/dense_13525/Tensordot/transpose	Transposeinput_13ai_model_6711/dense_13525/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
+ai_model_6711/dense_13525/Tensordot/ReshapeReshape1ai_model_6711/dense_13525/Tensordot/transpose:y:02ai_model_6711/dense_13525/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*ai_model_6711/dense_13525/Tensordot/MatMulMatMul4ai_model_6711/dense_13525/Tensordot/Reshape:output:0:ai_model_6711/dense_13525/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
+ai_model_6711/dense_13525/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:s
1ai_model_6711/dense_13525/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,ai_model_6711/dense_13525/Tensordot/concat_1ConcatV25ai_model_6711/dense_13525/Tensordot/GatherV2:output:04ai_model_6711/dense_13525/Tensordot/Const_2:output:0:ai_model_6711/dense_13525/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#ai_model_6711/dense_13525/TensordotReshape4ai_model_6711/dense_13525/Tensordot/MatMul:product:05ai_model_6711/dense_13525/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
0ai_model_6711/dense_13525/BiasAdd/ReadVariableOpReadVariableOp9ai_model_6711_dense_13525_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!ai_model_6711/dense_13525/BiasAddBiasAdd,ai_model_6711/dense_13525/Tensordot:output:08ai_model_6711/dense_13525/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
ai_model_6711/dense_13525/ReluRelu*ai_model_6711/dense_13525/BiasAdd:output:0*
T0*+
_output_shapes
:����������
#ai_model_6711/simple_rnn_6710/ShapeShape,ai_model_6711/dense_13525/Relu:activations:0*
T0*
_output_shapes
::��{
1ai_model_6711/simple_rnn_6710/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3ai_model_6711/simple_rnn_6710/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3ai_model_6711/simple_rnn_6710/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+ai_model_6711/simple_rnn_6710/strided_sliceStridedSlice,ai_model_6711/simple_rnn_6710/Shape:output:0:ai_model_6711/simple_rnn_6710/strided_slice/stack:output:0<ai_model_6711/simple_rnn_6710/strided_slice/stack_1:output:0<ai_model_6711/simple_rnn_6710/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,ai_model_6711/simple_rnn_6710/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
*ai_model_6711/simple_rnn_6710/zeros/packedPack4ai_model_6711/simple_rnn_6710/strided_slice:output:05ai_model_6711/simple_rnn_6710/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:n
)ai_model_6711/simple_rnn_6710/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
#ai_model_6711/simple_rnn_6710/zerosFill3ai_model_6711/simple_rnn_6710/zeros/packed:output:02ai_model_6711/simple_rnn_6710/zeros/Const:output:0*
T0*'
_output_shapes
:���������@�
,ai_model_6711/simple_rnn_6710/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
'ai_model_6711/simple_rnn_6710/transpose	Transpose,ai_model_6711/dense_13525/Relu:activations:05ai_model_6711/simple_rnn_6710/transpose/perm:output:0*
T0*+
_output_shapes
:����������
%ai_model_6711/simple_rnn_6710/Shape_1Shape+ai_model_6711/simple_rnn_6710/transpose:y:0*
T0*
_output_shapes
::��}
3ai_model_6711/simple_rnn_6710/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5ai_model_6711/simple_rnn_6710/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5ai_model_6711/simple_rnn_6710/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-ai_model_6711/simple_rnn_6710/strided_slice_1StridedSlice.ai_model_6711/simple_rnn_6710/Shape_1:output:0<ai_model_6711/simple_rnn_6710/strided_slice_1/stack:output:0>ai_model_6711/simple_rnn_6710/strided_slice_1/stack_1:output:0>ai_model_6711/simple_rnn_6710/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
9ai_model_6711/simple_rnn_6710/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
+ai_model_6711/simple_rnn_6710/TensorArrayV2TensorListReserveBai_model_6711/simple_rnn_6710/TensorArrayV2/element_shape:output:06ai_model_6711/simple_rnn_6710/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Sai_model_6711/simple_rnn_6710/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Eai_model_6711/simple_rnn_6710/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor+ai_model_6711/simple_rnn_6710/transpose:y:0\ai_model_6711/simple_rnn_6710/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���}
3ai_model_6711/simple_rnn_6710/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5ai_model_6711/simple_rnn_6710/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5ai_model_6711/simple_rnn_6710/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-ai_model_6711/simple_rnn_6710/strided_slice_2StridedSlice+ai_model_6711/simple_rnn_6710/transpose:y:0<ai_model_6711/simple_rnn_6710/strided_slice_2/stack:output:0>ai_model_6711/simple_rnn_6710/strided_slice_2/stack_1:output:0>ai_model_6711/simple_rnn_6710/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
Cai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpLai_model_6711_simple_rnn_6710_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
4ai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMulMatMul6ai_model_6711/simple_rnn_6710/strided_slice_2:output:0Kai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Dai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpMai_model_6711_simple_rnn_6710_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
5ai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAddBiasAdd>ai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul:product:0Lai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Eai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpNai_model_6711_simple_rnn_6710_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
6ai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1MatMul,ai_model_6711/simple_rnn_6710/zeros:output:0Mai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1ai_model_6711/simple_rnn_6710/simple_rnn_cell/addAddV2>ai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAdd:output:0@ai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
2ai_model_6711/simple_rnn_6710/simple_rnn_cell/ReluRelu5ai_model_6711/simple_rnn_6710/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
;ai_model_6711/simple_rnn_6710/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
-ai_model_6711/simple_rnn_6710/TensorArrayV2_1TensorListReserveDai_model_6711/simple_rnn_6710/TensorArrayV2_1/element_shape:output:06ai_model_6711/simple_rnn_6710/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���d
"ai_model_6711/simple_rnn_6710/timeConst*
_output_shapes
: *
dtype0*
value	B : �
6ai_model_6711/simple_rnn_6710/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������r
0ai_model_6711/simple_rnn_6710/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
#ai_model_6711/simple_rnn_6710/whileWhile9ai_model_6711/simple_rnn_6710/while/loop_counter:output:0?ai_model_6711/simple_rnn_6710/while/maximum_iterations:output:0+ai_model_6711/simple_rnn_6710/time:output:06ai_model_6711/simple_rnn_6710/TensorArrayV2_1:handle:0,ai_model_6711/simple_rnn_6710/zeros:output:06ai_model_6711/simple_rnn_6710/strided_slice_1:output:0Uai_model_6711/simple_rnn_6710/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lai_model_6711_simple_rnn_6710_simple_rnn_cell_matmul_readvariableop_resourceMai_model_6711_simple_rnn_6710_simple_rnn_cell_biasadd_readvariableop_resourceNai_model_6711_simple_rnn_6710_simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*=
body5R3
1ai_model_6711_simple_rnn_6710_while_body_75822919*=
cond5R3
1ai_model_6711_simple_rnn_6710_while_cond_75822918*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
Nai_model_6711/simple_rnn_6710/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
@ai_model_6711/simple_rnn_6710/TensorArrayV2Stack/TensorListStackTensorListStack,ai_model_6711/simple_rnn_6710/while:output:3Wai_model_6711/simple_rnn_6710/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0�
3ai_model_6711/simple_rnn_6710/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������
5ai_model_6711/simple_rnn_6710/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5ai_model_6711/simple_rnn_6710/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-ai_model_6711/simple_rnn_6710/strided_slice_3StridedSliceIai_model_6711/simple_rnn_6710/TensorArrayV2Stack/TensorListStack:tensor:0<ai_model_6711/simple_rnn_6710/strided_slice_3/stack:output:0>ai_model_6711/simple_rnn_6710/strided_slice_3/stack_1:output:0>ai_model_6711/simple_rnn_6710/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask�
.ai_model_6711/simple_rnn_6710/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
)ai_model_6711/simple_rnn_6710/transpose_1	TransposeIai_model_6711/simple_rnn_6710/TensorArrayV2Stack/TensorListStack:tensor:07ai_model_6711/simple_rnn_6710/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@�
2ai_model_6711/dense_13526/Tensordot/ReadVariableOpReadVariableOp;ai_model_6711_dense_13526_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0r
(ai_model_6711/dense_13526/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(ai_model_6711/dense_13526/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)ai_model_6711/dense_13526/Tensordot/ShapeShape-ai_model_6711/simple_rnn_6710/transpose_1:y:0*
T0*
_output_shapes
::��s
1ai_model_6711/dense_13526/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,ai_model_6711/dense_13526/Tensordot/GatherV2GatherV22ai_model_6711/dense_13526/Tensordot/Shape:output:01ai_model_6711/dense_13526/Tensordot/free:output:0:ai_model_6711/dense_13526/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3ai_model_6711/dense_13526/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.ai_model_6711/dense_13526/Tensordot/GatherV2_1GatherV22ai_model_6711/dense_13526/Tensordot/Shape:output:01ai_model_6711/dense_13526/Tensordot/axes:output:0<ai_model_6711/dense_13526/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)ai_model_6711/dense_13526/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(ai_model_6711/dense_13526/Tensordot/ProdProd5ai_model_6711/dense_13526/Tensordot/GatherV2:output:02ai_model_6711/dense_13526/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+ai_model_6711/dense_13526/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*ai_model_6711/dense_13526/Tensordot/Prod_1Prod7ai_model_6711/dense_13526/Tensordot/GatherV2_1:output:04ai_model_6711/dense_13526/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/ai_model_6711/dense_13526/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*ai_model_6711/dense_13526/Tensordot/concatConcatV21ai_model_6711/dense_13526/Tensordot/free:output:01ai_model_6711/dense_13526/Tensordot/axes:output:08ai_model_6711/dense_13526/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)ai_model_6711/dense_13526/Tensordot/stackPack1ai_model_6711/dense_13526/Tensordot/Prod:output:03ai_model_6711/dense_13526/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-ai_model_6711/dense_13526/Tensordot/transpose	Transpose-ai_model_6711/simple_rnn_6710/transpose_1:y:03ai_model_6711/dense_13526/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
+ai_model_6711/dense_13526/Tensordot/ReshapeReshape1ai_model_6711/dense_13526/Tensordot/transpose:y:02ai_model_6711/dense_13526/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*ai_model_6711/dense_13526/Tensordot/MatMulMatMul4ai_model_6711/dense_13526/Tensordot/Reshape:output:0:ai_model_6711/dense_13526/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
+ai_model_6711/dense_13526/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:s
1ai_model_6711/dense_13526/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,ai_model_6711/dense_13526/Tensordot/concat_1ConcatV25ai_model_6711/dense_13526/Tensordot/GatherV2:output:04ai_model_6711/dense_13526/Tensordot/Const_2:output:0:ai_model_6711/dense_13526/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#ai_model_6711/dense_13526/TensordotReshape4ai_model_6711/dense_13526/Tensordot/MatMul:product:05ai_model_6711/dense_13526/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
0ai_model_6711/dense_13526/BiasAdd/ReadVariableOpReadVariableOp9ai_model_6711_dense_13526_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!ai_model_6711/dense_13526/BiasAddBiasAdd,ai_model_6711/dense_13526/Tensordot:output:08ai_model_6711/dense_13526/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
!ai_model_6711/dense_13526/SigmoidSigmoid*ai_model_6711/dense_13526/BiasAdd:output:0*
T0*+
_output_shapes
:���������x
IdentityIdentity%ai_model_6711/dense_13526/Sigmoid:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp1^ai_model_6711/dense_13525/BiasAdd/ReadVariableOp3^ai_model_6711/dense_13525/Tensordot/ReadVariableOp1^ai_model_6711/dense_13526/BiasAdd/ReadVariableOp3^ai_model_6711/dense_13526/Tensordot/ReadVariableOpE^ai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAdd/ReadVariableOpD^ai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul/ReadVariableOpF^ai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1/ReadVariableOp$^ai_model_6711/simple_rnn_6710/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : 2d
0ai_model_6711/dense_13525/BiasAdd/ReadVariableOp0ai_model_6711/dense_13525/BiasAdd/ReadVariableOp2h
2ai_model_6711/dense_13525/Tensordot/ReadVariableOp2ai_model_6711/dense_13525/Tensordot/ReadVariableOp2d
0ai_model_6711/dense_13526/BiasAdd/ReadVariableOp0ai_model_6711/dense_13526/BiasAdd/ReadVariableOp2h
2ai_model_6711/dense_13526/Tensordot/ReadVariableOp2ai_model_6711/dense_13526/Tensordot/ReadVariableOp2�
Dai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAdd/ReadVariableOpDai_model_6711/simple_rnn_6710/simple_rnn_cell/BiasAdd/ReadVariableOp2�
Cai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul/ReadVariableOpCai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul/ReadVariableOp2�
Eai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1/ReadVariableOpEai_model_6711/simple_rnn_6710/simple_rnn_cell/MatMul_1/ReadVariableOp2J
#ai_model_6711/simple_rnn_6710/while#ai_model_6711/simple_rnn_6710/while:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
0__inference_ai_model_6711_layer_call_fn_75823639
input_1
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823491s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823635:($
"
_user_specified_name
75823633:($
"
_user_specified_name
75823631:($
"
_user_specified_name
75823629:($
"
_user_specified_name
75823627:($
"
_user_specified_name
75823625:($
"
_user_specified_name
75823623:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�=
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824126

inputs@
.simple_rnn_cell_matmul_readvariableop_resource:@=
/simple_rnn_cell_biasadd_readvariableop_resource:@B
0simple_rnn_cell_matmul_1_readvariableop_resource:@@
identity��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75824060*
condR
while_cond_75824059*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_75823067
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75823067___redundant_placeholder06
2while_while_cond_75823067___redundant_placeholder16
2while_while_cond_75823067___redundant_placeholder26
2while_while_cond_75823067___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�.
�
while_body_75823380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
6while_simple_rnn_cell_matmul_readvariableop_resource_0:@E
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:@J
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
4while_simple_rnn_cell_matmul_readvariableop_resource:@C
5while_simple_rnn_cell_biasadd_readvariableop_resource:@H
6while_simple_rnn_cell_matmul_1_readvariableop_resource:@@��,while/simple_rnn_cell/BiasAdd/ReadVariableOp�+while/simple_rnn_cell/MatMul/ReadVariableOp�-while/simple_rnn_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@s
while/simple_rnn_cell/ReluReluwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder(while/simple_rnn_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity(while/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�	
�
2__inference_simple_rnn_6710_layer_call_fn_75823769
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823131|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823765:($
"
_user_specified_name
75823763:($
"
_user_specified_name
75823761:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
0__inference_ai_model_6711_layer_call_fn_75823658
input_1
unknown:
	unknown_0:
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823620s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823654:($
"
_user_specified_name
75823652:($
"
_user_specified_name
75823650:($
"
_user_specified_name
75823648:($
"
_user_specified_name
75823646:($
"
_user_specified_name
75823644:($
"
_user_specified_name
75823642:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
I__inference_dense_13526_layer_call_and_return_conditional_losses_75823484

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�K
�
1ai_model_6711_simple_rnn_6710_while_body_75822919X
Tai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_loop_counter^
Zai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_maximum_iterations3
/ai_model_6711_simple_rnn_6710_while_placeholder5
1ai_model_6711_simple_rnn_6710_while_placeholder_15
1ai_model_6711_simple_rnn_6710_while_placeholder_2W
Sai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_strided_slice_1_0�
�ai_model_6711_simple_rnn_6710_while_tensorarrayv2read_tensorlistgetitem_ai_model_6711_simple_rnn_6710_tensorarrayunstack_tensorlistfromtensor_0f
Tai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_readvariableop_resource_0:@c
Uai_model_6711_simple_rnn_6710_while_simple_rnn_cell_biasadd_readvariableop_resource_0:@h
Vai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_1_readvariableop_resource_0:@@0
,ai_model_6711_simple_rnn_6710_while_identity2
.ai_model_6711_simple_rnn_6710_while_identity_12
.ai_model_6711_simple_rnn_6710_while_identity_22
.ai_model_6711_simple_rnn_6710_while_identity_32
.ai_model_6711_simple_rnn_6710_while_identity_4U
Qai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_strided_slice_1�
�ai_model_6711_simple_rnn_6710_while_tensorarrayv2read_tensorlistgetitem_ai_model_6711_simple_rnn_6710_tensorarrayunstack_tensorlistfromtensord
Rai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_readvariableop_resource:@a
Sai_model_6711_simple_rnn_6710_while_simple_rnn_cell_biasadd_readvariableop_resource:@f
Tai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_1_readvariableop_resource:@@��Jai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAdd/ReadVariableOp�Iai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul/ReadVariableOp�Kai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1/ReadVariableOp�
Uai_model_6711/simple_rnn_6710/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Gai_model_6711/simple_rnn_6710/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�ai_model_6711_simple_rnn_6710_while_tensorarrayv2read_tensorlistgetitem_ai_model_6711_simple_rnn_6710_tensorarrayunstack_tensorlistfromtensor_0/ai_model_6711_simple_rnn_6710_while_placeholder^ai_model_6711/simple_rnn_6710/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
Iai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpTai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
:ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMulMatMulNai_model_6711/simple_rnn_6710/while/TensorArrayV2Read/TensorListGetItem:item:0Qai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Jai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpUai_model_6711_simple_rnn_6710_while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
;ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAddBiasAddDai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul:product:0Rai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
Kai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpVai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
<ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1MatMul1ai_model_6711_simple_rnn_6710_while_placeholder_2Sai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
7ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/addAddV2Dai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAdd:output:0Fai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@�
8ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/ReluRelu;ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
Hai_model_6711/simple_rnn_6710/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem1ai_model_6711_simple_rnn_6710_while_placeholder_1/ai_model_6711_simple_rnn_6710_while_placeholderFai_model_6711/simple_rnn_6710/while/simple_rnn_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:���k
)ai_model_6711/simple_rnn_6710/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
'ai_model_6711/simple_rnn_6710/while/addAddV2/ai_model_6711_simple_rnn_6710_while_placeholder2ai_model_6711/simple_rnn_6710/while/add/y:output:0*
T0*
_output_shapes
: m
+ai_model_6711/simple_rnn_6710/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
)ai_model_6711/simple_rnn_6710/while/add_1AddV2Tai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_loop_counter4ai_model_6711/simple_rnn_6710/while/add_1/y:output:0*
T0*
_output_shapes
: �
,ai_model_6711/simple_rnn_6710/while/IdentityIdentity-ai_model_6711/simple_rnn_6710/while/add_1:z:0)^ai_model_6711/simple_rnn_6710/while/NoOp*
T0*
_output_shapes
: �
.ai_model_6711/simple_rnn_6710/while/Identity_1IdentityZai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_while_maximum_iterations)^ai_model_6711/simple_rnn_6710/while/NoOp*
T0*
_output_shapes
: �
.ai_model_6711/simple_rnn_6710/while/Identity_2Identity+ai_model_6711/simple_rnn_6710/while/add:z:0)^ai_model_6711/simple_rnn_6710/while/NoOp*
T0*
_output_shapes
: �
.ai_model_6711/simple_rnn_6710/while/Identity_3IdentityXai_model_6711/simple_rnn_6710/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^ai_model_6711/simple_rnn_6710/while/NoOp*
T0*
_output_shapes
: �
.ai_model_6711/simple_rnn_6710/while/Identity_4IdentityFai_model_6711/simple_rnn_6710/while/simple_rnn_cell/Relu:activations:0)^ai_model_6711/simple_rnn_6710/while/NoOp*
T0*'
_output_shapes
:���������@�
(ai_model_6711/simple_rnn_6710/while/NoOpNoOpK^ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAdd/ReadVariableOpJ^ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul/ReadVariableOpL^ai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "�
Qai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_strided_slice_1Sai_model_6711_simple_rnn_6710_while_ai_model_6711_simple_rnn_6710_strided_slice_1_0"i
.ai_model_6711_simple_rnn_6710_while_identity_17ai_model_6711/simple_rnn_6710/while/Identity_1:output:0"i
.ai_model_6711_simple_rnn_6710_while_identity_27ai_model_6711/simple_rnn_6710/while/Identity_2:output:0"i
.ai_model_6711_simple_rnn_6710_while_identity_37ai_model_6711/simple_rnn_6710/while/Identity_3:output:0"i
.ai_model_6711_simple_rnn_6710_while_identity_47ai_model_6711/simple_rnn_6710/while/Identity_4:output:0"e
,ai_model_6711_simple_rnn_6710_while_identity5ai_model_6711/simple_rnn_6710/while/Identity:output:0"�
Sai_model_6711_simple_rnn_6710_while_simple_rnn_cell_biasadd_readvariableop_resourceUai_model_6711_simple_rnn_6710_while_simple_rnn_cell_biasadd_readvariableop_resource_0"�
Tai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_1_readvariableop_resourceVai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_1_readvariableop_resource_0"�
Rai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_readvariableop_resourceTai_model_6711_simple_rnn_6710_while_simple_rnn_cell_matmul_readvariableop_resource_0"�
�ai_model_6711_simple_rnn_6710_while_tensorarrayv2read_tensorlistgetitem_ai_model_6711_simple_rnn_6710_tensorarrayunstack_tensorlistfromtensor�ai_model_6711_simple_rnn_6710_while_tensorarrayv2read_tensorlistgetitem_ai_model_6711_simple_rnn_6710_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2�
Jai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAdd/ReadVariableOpJai_model_6711/simple_rnn_6710/while/simple_rnn_cell/BiasAdd/ReadVariableOp2�
Iai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul/ReadVariableOpIai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul/ReadVariableOp2�
Kai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1/ReadVariableOpKai_model_6711/simple_rnn_6710/while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:}y

_output_shapes
: 
_
_user_specified_nameGEai_model_6711/simple_rnn_6710/TensorArrayUnstack/TensorListFromTensor:ea

_output_shapes
: 
G
_user_specified_name/-ai_model_6711/simple_rnn_6710/strided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :nj

_output_shapes
: 
P
_user_specified_name86ai_model_6711/simple_rnn_6710/while/maximum_iterations:h d

_output_shapes
: 
J
_user_specified_name20ai_model_6711/simple_rnn_6710/while/loop_counter
�
�
while_cond_75824167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75824167___redundant_placeholder06
2while_while_cond_75824167___redundant_placeholder16
2while_while_cond_75824167___redundant_placeholder26
2while_while_cond_75824167___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
while_cond_75823186
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75823186___redundant_placeholder06
2while_while_cond_75823186___redundant_placeholder16
2while_while_cond_75823186___redundant_placeholder26
2while_while_cond_75823186___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�=
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823446

inputs@
.simple_rnn_cell_matmul_readvariableop_resource:@=
/simple_rnn_cell_biasadd_readvariableop_resource:@B
0simple_rnn_cell_matmul_1_readvariableop_resource:@@
identity��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75823380*
condR
while_cond_75823379*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823491
input_1&
dense_13525_75823334:"
dense_13525_75823336:*
simple_rnn_6710_75823447:@&
simple_rnn_6710_75823449:@*
simple_rnn_6710_75823451:@@&
dense_13526_75823485:@"
dense_13526_75823487:
identity��#dense_13525/StatefulPartitionedCall�#dense_13526/StatefulPartitionedCall�'simple_rnn_6710/StatefulPartitionedCall�
#dense_13525/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_13525_75823334dense_13525_75823336*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823333�
'simple_rnn_6710/StatefulPartitionedCallStatefulPartitionedCall,dense_13525/StatefulPartitionedCall:output:0simple_rnn_6710_75823447simple_rnn_6710_75823449simple_rnn_6710_75823451*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823446�
#dense_13526/StatefulPartitionedCallStatefulPartitionedCall0simple_rnn_6710/StatefulPartitionedCall:output:0dense_13526_75823485dense_13526_75823487*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_13526_layer_call_and_return_conditional_losses_75823484
IdentityIdentity,dense_13526/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^dense_13525/StatefulPartitionedCall$^dense_13526/StatefulPartitionedCall(^simple_rnn_6710/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : 2J
#dense_13525/StatefulPartitionedCall#dense_13525/StatefulPartitionedCall2J
#dense_13526/StatefulPartitionedCall#dense_13526/StatefulPartitionedCall2R
'simple_rnn_6710/StatefulPartitionedCall'simple_rnn_6710/StatefulPartitionedCall:($
"
_user_specified_name
75823487:($
"
_user_specified_name
75823485:($
"
_user_specified_name
75823451:($
"
_user_specified_name
75823449:($
"
_user_specified_name
75823447:($
"
_user_specified_name
75823336:($
"
_user_specified_name
75823334:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823620
input_1&
dense_13525_75823494:"
dense_13525_75823496:*
simple_rnn_6710_75823607:@&
simple_rnn_6710_75823609:@*
simple_rnn_6710_75823611:@@&
dense_13526_75823614:@"
dense_13526_75823616:
identity��#dense_13525/StatefulPartitionedCall�#dense_13526/StatefulPartitionedCall�'simple_rnn_6710/StatefulPartitionedCall�
#dense_13525/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_13525_75823494dense_13525_75823496*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823333�
'simple_rnn_6710/StatefulPartitionedCallStatefulPartitionedCall,dense_13525/StatefulPartitionedCall:output:0simple_rnn_6710_75823607simple_rnn_6710_75823609simple_rnn_6710_75823611*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823606�
#dense_13526/StatefulPartitionedCallStatefulPartitionedCall0simple_rnn_6710/StatefulPartitionedCall:output:0dense_13526_75823614dense_13526_75823616*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_13526_layer_call_and_return_conditional_losses_75823484
IdentityIdentity,dense_13526/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp$^dense_13525/StatefulPartitionedCall$^dense_13526/StatefulPartitionedCall(^simple_rnn_6710/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : 2J
#dense_13525/StatefulPartitionedCall#dense_13525/StatefulPartitionedCall2J
#dense_13526/StatefulPartitionedCall#dense_13526/StatefulPartitionedCall2R
'simple_rnn_6710/StatefulPartitionedCall'simple_rnn_6710/StatefulPartitionedCall:($
"
_user_specified_name
75823616:($
"
_user_specified_name
75823614:($
"
_user_specified_name
75823611:($
"
_user_specified_name
75823609:($
"
_user_specified_name
75823607:($
"
_user_specified_name
75823496:($
"
_user_specified_name
75823494:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�G
�
!__inference__traced_save_75824400
file_prefixI
7read_disablecopyonread_ai_model_6711_dense_13525_kernel:E
7read_1_disablecopyonread_ai_model_6711_dense_13525_bias:_
Mread_2_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_kernel:@i
Wread_3_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_recurrent_kernel:@@Y
Kread_4_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_bias:@K
9read_5_disablecopyonread_ai_model_6711_dense_13526_kernel:@E
7read_6_disablecopyonread_ai_model_6711_dense_13526_bias:
savev2_const
identity_15��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOpw
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
: �
Read/DisableCopyOnReadDisableCopyOnRead7read_disablecopyonread_ai_model_6711_dense_13525_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp7read_disablecopyonread_ai_model_6711_dense_13525_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_1/DisableCopyOnReadDisableCopyOnRead7read_1_disablecopyonread_ai_model_6711_dense_13525_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp7read_1_disablecopyonread_ai_model_6711_dense_13525_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnReadMread_2_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOpMread_2_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_3/DisableCopyOnReadDisableCopyOnReadWread_3_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOpWread_3_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_4/DisableCopyOnReadDisableCopyOnReadKread_4_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpKread_4_disablecopyonread_ai_model_6711_simple_rnn_6710_simple_rnn_cell_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_5/DisableCopyOnReadDisableCopyOnRead9read_5_disablecopyonread_ai_model_6711_dense_13526_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp9read_5_disablecopyonread_ai_model_6711_dense_13526_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_6/DisableCopyOnReadDisableCopyOnRead7read_6_disablecopyonread_ai_model_6711_dense_13526_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp7read_6_disablecopyonread_ai_model_6711_dense_13526_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_14Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_15IdentityIdentity_14:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:>:
8
_user_specified_name ai_model_6711/dense_13526/bias:@<
:
_user_specified_name" ai_model_6711/dense_13526/kernel:RN
L
_user_specified_name42ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias:^Z
X
_user_specified_name@>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel:TP
N
_user_specified_name64ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel:>:
8
_user_specified_name ai_model_6711/dense_13525/bias:@<
:
_user_specified_name" ai_model_6711/dense_13525/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�=
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824234

inputs@
.simple_rnn_cell_matmul_readvariableop_resource:@=
/simple_rnn_cell_biasadd_readvariableop_resource:@B
0simple_rnn_cell_matmul_1_readvariableop_resource:@@
identity��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75824168*
condR
while_cond_75824167*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
while_body_75823068
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_75823090_0:@.
 while_simple_rnn_cell_75823092_0:@2
 while_simple_rnn_cell_75823094_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_75823090:@,
while_simple_rnn_cell_75823092:@0
while_simple_rnn_cell_75823094:@@��-while/simple_rnn_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_75823090_0 while_simple_rnn_cell_75823092_0 while_simple_rnn_cell_75823094_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823055�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder6while/simple_rnn_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity6while/simple_rnn_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@X

while/NoOpNoOp.^while/simple_rnn_cell/StatefulPartitionedCall*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"B
while_simple_rnn_cell_75823090 while_simple_rnn_cell_75823090_0"B
while_simple_rnn_cell_75823092 while_simple_rnn_cell_75823092_0"B
while_simple_rnn_cell_75823094 while_simple_rnn_cell_75823094_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2^
-while/simple_rnn_cell/StatefulPartitionedCall-while/simple_rnn_cell/StatefulPartitionedCall:(	$
"
_user_specified_name
75823094:($
"
_user_specified_name
75823092:($
"
_user_specified_name
75823090:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824336

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
ReluReluadd:z:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:���������@
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823910
inputs_0@
.simple_rnn_cell_matmul_readvariableop_resource:@=
/simple_rnn_cell_biasadd_readvariableop_resource:@B
0simple_rnn_cell_matmul_1_readvariableop_resource:@@
identity��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75823844*
condR
while_cond_75823843*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@�
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�	
�
2__inference_simple_rnn_6710_layer_call_fn_75823780
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823250|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75823776:($
"
_user_specified_name
75823774:($
"
_user_specified_name
75823772:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�=
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823606

inputs@
.simple_rnn_cell_matmul_readvariableop_resource:@=
/simple_rnn_cell_biasadd_readvariableop_resource:@B
0simple_rnn_cell_matmul_1_readvariableop_resource:@@
identity��&simple_rnn_cell/BiasAdd/ReadVariableOp�%simple_rnn_cell/MatMul/ReadVariableOp�'simple_rnn_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0�
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@g
simple_rnn_cell/ReluRelusimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75823540*
condR
while_cond_75823539*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823758

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_simple_rnn_cell_layer_call_fn_75824302

inputs
states_0
unknown:@
	unknown_0:@
	unknown_1:@@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
75824296:($
"
_user_specified_name
75824294:($
"
_user_specified_name
75824292:QM
'
_output_shapes
:���������@
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_75823951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_75823951___redundant_placeholder06
2while_while_cond_75823951___redundant_placeholder16
2while_while_cond_75823951___redundant_placeholder26
2while_while_cond_75823951___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: :::::

_output_shapes
::GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�.
�
while_body_75823952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
6while_simple_rnn_cell_matmul_readvariableop_resource_0:@E
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:@J
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
4while_simple_rnn_cell_matmul_readvariableop_resource:@C
5while_simple_rnn_cell_biasadd_readvariableop_resource:@H
6while_simple_rnn_cell_matmul_1_readvariableop_resource:@@��,while/simple_rnn_cell/BiasAdd/ReadVariableOp�+while/simple_rnn_cell/MatMul/ReadVariableOp�-while/simple_rnn_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@s
while/simple_rnn_cell/ReluReluwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder(while/simple_rnn_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity(while/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�
�
I__inference_dense_13526_layer_call_and_return_conditional_losses_75824274

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�4
�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823250

inputs*
simple_rnn_cell_75823175:@&
simple_rnn_cell_75823177:@*
simple_rnn_cell_75823179:@@
identity��'simple_rnn_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_75823175simple_rnn_cell_75823177simple_rnn_cell_75823179*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75823174n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_75823175simple_rnn_cell_75823177simple_rnn_cell_75823179*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_75823187*
condR
while_cond_75823186*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@T
NoOpNoOp(^simple_rnn_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2R
'simple_rnn_cell/StatefulPartitionedCall'simple_rnn_cell/StatefulPartitionedCall2
whilewhile:($
"
_user_specified_name
75823179:($
"
_user_specified_name
75823177:($
"
_user_specified_name
75823175:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824319

inputs
states_00
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������@G
ReluReluadd:z:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
'
_output_shapes
:���������@
"
_user_specified_name
states_0:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
while_body_75823540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
6while_simple_rnn_cell_matmul_readvariableop_resource_0:@E
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:@J
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
4while_simple_rnn_cell_matmul_readvariableop_resource:@C
5while_simple_rnn_cell_biasadd_readvariableop_resource:@H
6while_simple_rnn_cell_matmul_1_readvariableop_resource:@@��,while/simple_rnn_cell/BiasAdd/ReadVariableOp�+while/simple_rnn_cell/MatMul/ReadVariableOp�-while/simple_rnn_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@s
while/simple_rnn_cell/ReluReluwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder(while/simple_rnn_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity(while/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�.
�
while_body_75824168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
6while_simple_rnn_cell_matmul_readvariableop_resource_0:@E
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:@J
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
4while_simple_rnn_cell_matmul_readvariableop_resource:@C
5while_simple_rnn_cell_biasadd_readvariableop_resource:@H
6while_simple_rnn_cell_matmul_1_readvariableop_resource:@@��,while/simple_rnn_cell/BiasAdd/ReadVariableOp�+while/simple_rnn_cell/MatMul/ReadVariableOp�-while/simple_rnn_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@s
while/simple_rnn_cell/ReluReluwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder(while/simple_rnn_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity(while/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
�.
�
while_body_75823844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
6while_simple_rnn_cell_matmul_readvariableop_resource_0:@E
7while_simple_rnn_cell_biasadd_readvariableop_resource_0:@J
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
4while_simple_rnn_cell_matmul_readvariableop_resource:@C
5while_simple_rnn_cell_biasadd_readvariableop_resource:@H
6while_simple_rnn_cell_matmul_1_readvariableop_resource:@@��,while/simple_rnn_cell/BiasAdd/ReadVariableOp�+while/simple_rnn_cell/MatMul/ReadVariableOp�-while/simple_rnn_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0�
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0�
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0�
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:���������@s
while/simple_rnn_cell/ReluReluwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:���������@�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder(while/simple_rnn_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity(while/simple_rnn_cell/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0")
while_identitywhile/Identity:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������@
output_14
StatefulPartitionedCall:0���������tensorflow/serving/predict:֑
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1
	rnn


dense2

signatures"
_tf_keras_model
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
0__inference_ai_model_6711_layer_call_fn_75823639
0__inference_ai_model_6711_layer_call_fn_75823658�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823491
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823620�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�B�
#__inference__wrapped_model_75823012input_1"�
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
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(cell
)
state_spec"
_tf_keras_rnn_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
,
0serving_default"
signature_map
2:02 ai_model_6711/dense_13525/kernel
,:*2ai_model_6711/dense_13525/bias
F:D@24ai_model_6711/simple_rnn_6710/simple_rnn_cell/kernel
P:N@@2>ai_model_6711/simple_rnn_6710/simple_rnn_cell/recurrent_kernel
@:>@22ai_model_6711/simple_rnn_6710/simple_rnn_cell/bias
2:0@2 ai_model_6711/dense_13526/kernel
,:*2ai_model_6711/dense_13526/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_ai_model_6711_layer_call_fn_75823639input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
0__inference_ai_model_6711_layer_call_fn_75823658input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823491input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823620input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
6trace_02�
.__inference_dense_13525_layer_call_fn_75823727�
���
FullArgSpec
args�

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
annotations� *
 z6trace_0
�
7trace_02�
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823758�
���
FullArgSpec
args�

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
annotations� *
 z7trace_0
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

8states
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
>trace_0
?trace_1
@trace_2
Atrace_32�
2__inference_simple_rnn_6710_layer_call_fn_75823769
2__inference_simple_rnn_6710_layer_call_fn_75823780
2__inference_simple_rnn_6710_layer_call_fn_75823791
2__inference_simple_rnn_6710_layer_call_fn_75823802�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>trace_0z?trace_1z@trace_2zAtrace_3
�
Btrace_0
Ctrace_1
Dtrace_2
Etrace_32�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823910
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824018
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824126
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824234�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zBtrace_0zCtrace_1zDtrace_2zEtrace_3
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_random_generator

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_02�
.__inference_dense_13526_layer_call_fn_75824243�
���
FullArgSpec
args�

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
annotations� *
 zRtrace_0
�
Strace_02�
I__inference_dense_13526_layer_call_and_return_conditional_losses_75824274�
���
FullArgSpec
args�

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
annotations� *
 zStrace_0
�B�
&__inference_signature_wrapper_75823718input_1"�
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_dense_13525_layer_call_fn_75823727inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823758inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_simple_rnn_6710_layer_call_fn_75823769inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_simple_rnn_6710_layer_call_fn_75823780inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_simple_rnn_6710_layer_call_fn_75823791inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_simple_rnn_6710_layer_call_fn_75823802inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823910inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824018inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824126inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824234inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
Ytrace_0
Ztrace_12�
2__inference_simple_rnn_cell_layer_call_fn_75824288
2__inference_simple_rnn_cell_layer_call_fn_75824302�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0zZtrace_1
�
[trace_0
\trace_12�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824319
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824336�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0z\trace_1
"
_generic_user_object
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
.__inference_dense_13526_layer_call_fn_75824243inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
I__inference_dense_13526_layer_call_and_return_conditional_losses_75824274inputs"�
���
FullArgSpec
args�

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
annotations� *
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
�B�
2__inference_simple_rnn_cell_layer_call_fn_75824288inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_simple_rnn_cell_layer_call_fn_75824302inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824319inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824336inputsstates_0"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_75823012x4�1
*�'
%�"
input_1���������
� "7�4
2
output_1&�#
output_1����������
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823491�D�A
*�'
%�"
input_1���������
�

trainingp"0�-
&�#
tensor_0���������
� �
K__inference_ai_model_6711_layer_call_and_return_conditional_losses_75823620�D�A
*�'
%�"
input_1���������
�

trainingp "0�-
&�#
tensor_0���������
� �
0__inference_ai_model_6711_layer_call_fn_75823639vD�A
*�'
%�"
input_1���������
�

trainingp"%�"
unknown����������
0__inference_ai_model_6711_layer_call_fn_75823658vD�A
*�'
%�"
input_1���������
�

trainingp "%�"
unknown����������
I__inference_dense_13525_layer_call_and_return_conditional_losses_75823758k3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
.__inference_dense_13525_layer_call_fn_75823727`3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
I__inference_dense_13526_layer_call_and_return_conditional_losses_75824274k3�0
)�&
$�!
inputs���������@
� "0�-
&�#
tensor_0���������
� �
.__inference_dense_13526_layer_call_fn_75824243`3�0
)�&
$�!
inputs���������@
� "%�"
unknown����������
&__inference_signature_wrapper_75823718�?�<
� 
5�2
0
input_1%�"
input_1���������"7�4
2
output_1&�#
output_1����������
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75823910�O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "9�6
/�,
tensor_0������������������@
� �
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824018�O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "9�6
/�,
tensor_0������������������@
� �
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824126x?�<
5�2
$�!
inputs���������

 
p

 
� "0�-
&�#
tensor_0���������@
� �
M__inference_simple_rnn_6710_layer_call_and_return_conditional_losses_75824234x?�<
5�2
$�!
inputs���������

 
p 

 
� "0�-
&�#
tensor_0���������@
� �
2__inference_simple_rnn_6710_layer_call_fn_75823769�O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ".�+
unknown������������������@�
2__inference_simple_rnn_6710_layer_call_fn_75823780�O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ".�+
unknown������������������@�
2__inference_simple_rnn_6710_layer_call_fn_75823791m?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
unknown���������@�
2__inference_simple_rnn_6710_layer_call_fn_75823802m?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
unknown���������@�
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824319�\�Y
R�O
 �
inputs���������
'�$
"�
states_0���������@
p
� "`�]
V�S
$�!

tensor_0_0���������@
+�(
&�#
tensor_0_1_0���������@
� �
M__inference_simple_rnn_cell_layer_call_and_return_conditional_losses_75824336�\�Y
R�O
 �
inputs���������
'�$
"�
states_0���������@
p 
� "`�]
V�S
$�!

tensor_0_0���������@
+�(
&�#
tensor_0_1_0���������@
� �
2__inference_simple_rnn_cell_layer_call_fn_75824288�\�Y
R�O
 �
inputs���������
'�$
"�
states_0���������@
p
� "R�O
"�
tensor_0���������@
)�&
$�!

tensor_1_0���������@�
2__inference_simple_rnn_cell_layer_call_fn_75824302�\�Y
R�O
 �
inputs���������
'�$
"�
states_0���������@
p 
� "R�O
"�
tensor_0���������@
)�&
$�!

tensor_1_0���������@