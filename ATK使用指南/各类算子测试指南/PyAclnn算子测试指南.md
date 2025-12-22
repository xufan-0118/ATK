# PyAclnn算子测试指南

[toc]

---

# 环境准备

## 安装ATK

> 若要测试自定义算子需确保已经成功部署到测试环境中，如部署自定义算子包后会自动生成aclnn调用文件。
> 同时需确保标杆算子能够正常调用，如cann包自带算子需安装torch_npu包。

本工具主要为用户提供软件包安装方式，用户获取到安装包后，上传至服务器并执行如下命令安装工具

- 软件包安装工具命令

```shell
pip install ATK*.whl
```

执行下面命令回显正常表示安装成功

```sehll
pip show atk
```

## 配置测试算子so路径

pyaclnn按照如下读取顺序在环境中寻找

1. 在pyaclnn执行时需获取aclnn算子的函数所在的so路径。读取顺序如下，若三个路径均不存在，所调算子会报错：
   ```python
   ${ASCEND_CUSTOM_OPP_PATH}/vendors/customize/op_api/lib/libcust_opapi.so
   ${ASCEND_OPP_PATH}/vendors/customize/op_api/lib/libcust_opapi.so
   ${ASCEND_OPP_PATH}/lib64/libopapi.so
   ```
2. 在pyaclnn执行时需读取依赖库，默认路径如下，若自定义需设置环境变量`ASCEND_TOOLKIT_HOME`：
   ```python
   # 配置库路径
   ASCEND_TOOLKIT_HOME = os.environ.get('ASCEND_TOOLKIT_HOME')
   if ASCEND_TOOLKIT_HOME is None:
       ASCEND_TOOLKIT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
   ACLNN_FUNC_PATH = get_opp_lib_path()
   ASCENDCL_PATH = os.path.join(ASCEND_TOOLKIT_HOME, "acllib/lib64/libascendcl.so")
   NNOPBASE_PATH = os.path.join(ASCEND_TOOLKIT_HOME, "acllib/lib64/libnnopbase.so")
   ```

# 测试用例生成

算子测试的核心输入之一就是算子用例，为了保障算子满足各类使用场景，算子用例需要具备较强的泛化性，覆盖各类shape、数据类型和数据分布等。ATK工具可以根据算子入参的各类场景，对输入Tensor、属性参数进行泛化组合，自动生成符合要求的测试用例集。

## 用例设计

在进行用例生成之前，需要设计用例生成的各类参数的yaml文件，atk工具通过解析文件信息自动生成用例集，以aclnnMaxDim算子为例，yaml文件格式及说明可参考以下内容。
测试aclnn算子，inputs参数的所有输入需严格按照aclnn的输入顺序填写，cann包自带算子可参考[aclnnMaxDim开发文档-昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/apiref/appdevgapi/context/aclnnMaxDim.md)。

* 如`aclnnMaxDim`算子，两段式接口中的`aclnnMaxDimGetWorkspaceSize`定义为`aclStatus aclnnMaxDimGetWorkspaceSize(const aclTensor *self, const int64_t dim, const bool keepdim, aclTensor *out, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor)`
  此时输入顺序依次为`self`、`dim`、`keepdim`（注：输入名称可自定义）
* 当部署自定义算子包时，会自动生成aclnn调用文件，可通过查看文件`{ASCEND_CUSTOM_OPP_PATH}/vendors/customize/op_api/include/aclnn_*.h`确认aclnn算子的输入顺序，其中`ASCEND_CUSTOM_OPP_PATH`为自定义算子包部署的路径。

> 可自定义用例设计文件的存放位置，在用例生成时以绝对路径进行读取

下面是一个用例生成yaml的示例：

```yaml
# torch.max.yaml
api: pytorch
api_type: function
aclnn_api_type: aclnn_function
version: v2.1
name: torch.max
aclnn_name: MaxDim
dtype_numbers: 200
extra_numbers: all
generate: reduce
standard:
    acc: single_bm
    perf: not_key
inputs:
  - name: input
    type: tensor
    required: true
    dtypes:
      values: [ fp32, fp16 ]
    ranges:
      valid:
        values: [ [-5, 5] ]
      invalid:
        values: [ [-5, 5] ]
    shapes:
      dim_numbers:
        values: [1, 2, 3, 4, 5, 6, 7, 8]
      max_length: 4294967295
    boundary:
      has_empty: true
      has_infnan: true
      has_scalar: true
      has_upper_border: true
      has_lower_border: true
  - name: dim
    type: attr
    required: true
    dtypes:
      values: [ int ]
    ranges:
      valid:
        values: [ [-5, 5] ]
      invalid:
        values: [ [-5, 5] ]
  - name: keepdim
    type: attr
    required: false
    dtypes:
      values: [ bool ]
    ranges:
      valid:
        values: [ true,  false ]
        weights: [ 0.5, 0.5 ]
```

## 参数约束（可选）

当算子API的输入包含多个参数时，不同输入参数之间有可能存在依赖关系。此时由于工具无法准确获悉其依赖情况，导致泛化组合生成的测试用例存在问题，出现算子执行异常。因此需要用户对生成的用例进行合理修正，确保用例满足要求。
本工具提供了统一的插件入口，为用户实现较为复杂的用例约束场景。以测试aclnnMaxDim算子为例，参数dim对输入tensor有依赖关系，dim参数需要reduce的维度信息，取值范围需要小于tensor输入参数的维度，实现接口参考如下：

```python
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

@GENERATOR_REGISTRY.register("reduce")
class ReduceGenerator(CaseGenerator):

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        '''
        用例参数约束修改入口
        :param case_config:  生成的用例信息，可能不满足参数间约束，导致用例无效
        :return: 返回修改后符合参数间约束关系的用例，需要用例保障用例有效
        '''
        dim = len(case_config.inputs[0].shape)  # 获取第一个tensor参数shape最大维度值
        range_is_null = case_config.inputs[0].is_range_null()  # 判断是否为空tensor
        if range_is_null:
            case_config.inputs[1].range_values = [0]  # 空tensor设置维度值为0
        else:
            case_config.inputs[1].range_values = [random.randint(-dim, max(0, dim - 1))]  # 非空tensor设置dim在可选范围内随机
        return case_config  # 返回修改和符合参数约束的用例
```
> 在yaml文件中`generate`字段，将`default`修改为注册名称`reduce`

## 命令执行

用户设计完用例参数yaml文件后，可以通过ATK工具生成泛化用例，操作命令参考如下，其中torch.max.yaml是aclnnMaxDim算子的用例设计文件，可自定义用例设计文件名称，该方式会按照默认的数据生成规则来生成输入数据：

```shell
atk case -f atk/tests/torch.max.yaml
```

用例结果会保存在`result/torch.max/json/all_torch.max.json`下

更多使用参数说明可执行`atk case --help`查看

- 如需使用参数约束插件，可执行以下命令生成测试用例，该方式会通过自定义生成规则`reduce`来生成输入数据，`-p`入参为文件路径，表示自定义用例生成规则：

```shell
atk case -f atk/tests/torch.max.yaml -p atk/case_generator/generator/generate_types/generate_reduce.py
```



# 自定义api编写


调用aclnn算子接口时，需要调整输入或输出数据、修改标杆执行流程、修改标杆生成逻辑等场景时，可以通过新增文件继承基类来实现自己的标杆自定义执行方式。

详细的自定义api编写请参考：[PyAclnn自定义Api编写指南](./PyAclnn自定义API编写指南.md)



# 参数校验

**在执行前，需要检查输入参数的数据类型是否符合aclnn接口的定义，否则可能导致工具执行卡死！！！**

函数入参签名校验方法如下：

- 方式1

执行命令中加入参数`-cp`

```shell
atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/torch.max/json/all_torch.max.json --task accuracy -s 0 -e 1 -cp
```

- 方式2

查看aclnn算子定义，或者通过头文件查看
假设要测的aclnn算子头文件是`aclnn_max_dim.h`，对应算子aclnnMaxDim，执行以下命令:

```shell
find / -name "aclnn_max_dim.h"
```

文件中对aclnn算子的定义如下：

```python
ACLNN_API aclnnStatus aclnnMaxDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepdim,
                                                  aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);
```

则在自定义api文件中增加以下函数来实现函数入参签名校验：

```python
@register("aclnn_cpu_max")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu" or self.device == "npu":
            output = torch.max(
                *input_data.args, **input_data.kwargs
            )
        return output

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnMaxDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)"
```



# 精度测试

> 精度比对场景下，待测算子为aclnn，标杆通常运行在cpu。

执行下面命令进行aclnnMaxDim算子和cpu上的torch.max算子的精度比对

```shell
atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/torch.max/json/all_torch.max.json --task accuracy -s 0 -e 1
```

若需自定义标杆执行逻辑，可通过以下方式实现，其中`/home/aclnn_max.py`为自定义标杆执行逻辑的文件路径：

```shell
atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/torch.max/json/all_torch.max.json -p /home/aclnn_max.py --task accuracy -s 0 -e 1
```

执行命令及其可选参数作用介绍如下：

| 参数| 子参数 | 说明 |
| --- | --- | --- |
| node | --backend | 必选参数，表示执行后端，可选pyaclnn/cpu/npu |
| node  | --devices | 当backend为pyaclnn/npu时必选，表示使用的设备id |
| node  | --is_compare | 是否用来做比较，可选True/False |
| task | -c | 表示待测试的用例json文件 |
| task | --task | 表示执行的任务类型，可选accuracy/accuracy_dc/performance_device，分别表示精度比对/确定性计算/device性能，多个任务以逗号隔开 |
| task | -s | 表示执行的起始用例id  |
| task | -e | 表示执行的结束用例id（不包含）  |
| task | -p | 入参为文件路径，表示自定义标杆执行逻辑。同时在用例生成时要将yaml文件中的`api_type`字段修改为自定义标杆文件的注册器名称  |

执行完成后，会输出整体结果如下，同时也在output_test目录中会生成excel报告

```
+-------+----------+------------+--------------+
|  名称 | 总用例数  | 精度通过率 | 精度是否达标  |
+-------+----------+------------+--------------+
| cpu_0 |    1     |   100.0    |     Pass     |
+-------+----------+------------+--------------+
```

> 报告中显示全部用例精度或性能Pass即为测试通过，否则请查看excel文件中不通过算子的详细原因


# 性能测试

性能比对需要测试算子和标杆算子均需要在npu设备上进行测试，且同一个算子不能在同一个npu设备上初始化两次。

> 该场景下测试算子通常为aclnn，标杆算子为npu或aclnn。

## aclnn VS npu 性能比对

> 该场景下要挂载两张卡

执行下面命令进行aclnnMaxDim算子和npu上的torch.max算子的性能比对（进行device侧性能比较）：

```shell
atk node --backend pyaclnn --devices 0 node --backend npu --devices 1 task -c result/torch.max/json/all_torch.max.json --task performance_device -s 0 -e 1
```

执行结果如下所示：

```
+-------+----------+------------------+------------------+--------------------+
|  名称 | 总用例数  | device性能通过率 | 平均device性能比  | device性能是否达标  |
+-------+----------+------------------+------------------+--------------------+
| npu_1 |    1     |      100.0       |      0.5663      |        Pass        |
+-------+----------+------------------+------------------+--------------------+
```

## aclnn VS aclnn 性能比对

当对aclnn算子进行优化后，需对比前后两次的性能，可通过以下方式进行性能测试。

1.先保存优化前aclnn算子的性能数据。

```bash
atk node --backend pyaclnn --devices 0 --name aclnn_base node --backend cpu task -c result/torch.max/json/all_torch.max.json --task performance_device -s 0 -e 1
```

得到excel文件，假如保存到路径output_test/all_torch.max_reports_2025-02-19-14-17-42.xlsx


2.测试优化后aclnn算子的性能数据，将优化前的算子性能数据作为基线，进行优化后的算子性能测试。

```bash
atk node --backend pyaclnn --devices 0 node --backend cpu --is_compare False node --backend pyaclnn --devices 0 --bm_file output_test/all_torch.max_reports_2025-02-19-14-17-42.xlsx task -c result/torch.max/json/all_torch.max.json --task performance_device  -s 0 -e 1
```

执行参数含义如下：

| 参数| 子参数 | 说明 |
| --- | --- | --- |
| node | --name| 执行后端节点名称 |
| node | --bm_file | 标杆性能测试数据的excel文件路径，用于和已有性能数据比对 |

执行结果如下：

```
+--------------------+----------+------------------+------------------+--------------------+
|        名称        |  总用例数 | device性能通过率  | 平均device性能比 | device性能是否达标  |
+--------------------+----------+------------------+------------------+--------------------+
| pyaclnn_aclnn_base |    1     |      100.0       |      0.9848      |        Pass        |
+--------------------+----------+------------------+------------------+--------------------+
```

# 确定性计算测试

确定性计算测试是用于aclnn算子多次执行，确保输出一致的测试，以验证在输入数据集等输入条件不变的情况下，算子多次运行且输出结果每次保持一致，主要通过`-task accuracy_dc`参数进行控制，命令示例如下：

```shell
atk node --backend pyaclnn --devices 0  node --backend cpu --is_compare False task -c result/torch.max/json/all_torch.max.json --task accuracy_dc -e 400
```
