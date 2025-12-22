
# 算子Fuzz压测使用指南

[toc]

---

> 算子压测旨在通过控制用例的生成个数来生成大量的算子用例，然后一次在npu上执行用例直到用例执行失败或者用例结束。

# 配置算子用例yaml

该步骤与[用例生成](../用例生成.md)章节保持一致，需要注意：

* 由于该场景下会生成大量的用例，因此为了避免过多重复的用例，建议在shape、range等参数设计上添加更多的数值。

下面为torch.max的yaml样例：

```yaml
api: pytorch
api_type: function
version: v2.1
name: torch.max
aclnn_name: MaxDim
generate: reduce
backward: false
inputs:
  - name: input
    type: tensor
    required: true
    dtypes:
      values: [ fp32, fp16 ]
    ranges:
      valid:
        values: [ [-5, 5] ]
    shapes:
      dim_numbers:
        values: [1, 2, 3, 4, 5, 6, 7, 8]
      max_length: 4294967295
  - name: dim
    type: attr
    required: true
    dtypes:
      values: [ int ]
    ranges:
      valid:
        values: [ [ -7,7 ] , -8, 8, 9 ,'-inf', 'inf' ]
        weights: [ 0.8, 0.05, 0.05, 0.05, 0.05 , 0.05 ]
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

# 用例执行

* 因为只需在npu上验证，因此只需配置一个npu node即可
* 无需跑精度、性能任务
* 若想要与cpu、gpu跑精度、性能任务，task命令配置成相应的任务场景也是可以使用。

> 执行时不需要再单独执行用例生成阶段，直接将用例的yaml文件作为输入即可。

执行命令如下：

```shell
atk node --backend npu --devices 0 fuzz -f torch.max.yaml --fuzz_num 2
```

> fuzz_num用于设置用例的数目，默认为100000

```shell
atk node --backend npu --devices 0 fuzz -f torch.max.yaml --fuzz_time 1m --fuzz_time_dtype_num 2
```

> fuzz_time用于设置用例执行的时间，格式为[5m, 5h, 5d]，其中fuzz_time的优先级大于fuzz_num，
> 设置fuzz_time时会至少执行1轮，每一轮的样例数量为dtype个数*`fuzz_time_dtype_num`，
> 再判断执行时间是否超出fuzz_time，若跑测成功后仍有剩余时间，则会重新执行1轮，直到没有剩余时间，所以可能生成多个json和excel用例，**且每一轮生成的用例都会完全执行**。

# 输出结果

若出现以下语句（一般在打屏日志的最后），则压测成功；若没出现，一定没有成功，需检查。

```shell
run fuzz task success!
finish fuzz task
```

**用例**

压测的输出用例json和excel保存在`atk_output/{task*}/fuzz_op`目录下。

**报告**

输出至`atk_output/{task*}/excel`目录下，若设置fuzz_time, 可能有多个输出报告。

# 参数说明

`atk fuzz`的可选参数如下：

| 配置项                | 说明                                                         | 示例                                                         |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| --fuzz_num            | **选填**，设置算子fuzz测试用例数量，<br> 实际生成dtype个数*`fuzz_num`，默认为100000 | `atk node --backend npu --devices 0 fuzz -f torch.max.yaml --fuzz_num 2` |
| --fuzz_time           | **选填**，设置算子fuzz测试用例的执行执行, 结尾必须是m、 h、 d， <br>分别代表分、时、天, 如5m, 5h, 5d。  <br>其中fuzz_time参数优先级大于`fuzz_num`，即`fuzz_time`设置时`fuzz_num`失效，<br>生成的用例个数取决于`fuzz_time_dtype_num`，<br>且默认经过一轮计算时间，一轮的用例个数为dtype个数*`fuzz_time_dtype_num`，<br>即在执行完一轮任务后才会判断`fuzz_time` | `atk node --backend npu --devices 0 fuzz -f torch.max.yaml --fuzz_time 1m` |
| --fuzz_time_dtype_num | **选填**，和fuzz_time同时使用，设置fuzz_time后，可以定义每轮的用例数量，<br> 实际生成dtype个数*`fuzz_time_dtype_num`，默认为10000 | `atk node --backend npu --devices 0 fuzz -f torch.max.yaml --fuzz_time 1m --fuzz_time_dtype_num 2` |
