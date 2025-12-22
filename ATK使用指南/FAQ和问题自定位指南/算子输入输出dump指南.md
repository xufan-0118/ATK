# 算子输入输出dump指南

[toc]

---

API 脚本验证中会调用到多个算子，可以 dump 出链路中单算子输入输出数据和标杆比对，以验证单算子精度

#### dump文件生成

1. 环境变量设置：

```shell
export ACL_DUMP_DATA=1
```

2. 在脚本运行目录下放入 **acl.json** 文件

```json
{
    "dump":{
        "dump_path":"/home/dump",
        "dump_mode": "all",
        "dump_debug": "off",
        "dump_op_switch": "on"
    }
}
```

注意:
(1) dump_path 为自定义路径，需要提前建立好，权限正确
(2) 上述配置默认是 dump 调用到所有的算子的所有输入输出。需要精细化控制，需要修改 json 的配置。

3. 运行单算子或者网络脚本， 在 /home/dump 中生成 dump 文件。

#### dump文件格式说明

1. 进入自定义的 dump_path，示例中为 /home/dump，进入脚本对应输出目录

```shell
cd /home/dump/20230701142635/0/0/
```

2. 得到的文件名包含算子名称及调用顺序

> ResizeBilinearV2D.ResizeBilinearV2.4.2.1688192805720118
> TransData.trans_TransData_0.2.2.1688192805706027
> TransData.trans_TransData_1.6.2.1688192810282034

#### dump文件查看方式

通过如下命令，执行CANN包中的msaccucmp.py脚本生成 npy 文件：

```shell

python3 /usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py convert -d dump算子文件 -out 输出目录 -v 2
```

得到对应 ResizeBilinearV2 算子的输入输出文件

> ResizeBilinearV2.4.2.1688192805720118.input.0.npy
> ResizeBilinearV2.4.2.1688192805720118.output.0.npy



