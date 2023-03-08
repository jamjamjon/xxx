[x] 1. Base() class : batch infer , pt, engine model support
[x] 2. ModelManager() class: track model class instances
[x] 3. RequestSolver()
# AllRounder class  = ModelManager() + RequestSolver()



# TODO
[ ] serve engine solve
[ ] numba speed up 
[ ] model visualization
[ ] 用于多线程的多模型维护容器
[ ] base config 新增，作为基类配置，用于其他配置继承，重新优化配置, 包括component name
[ ] 配置文件中的`module`字段可以修改成为component name!
[x] rename base.py base_model.py
[ ] IObingding: has synchronize_input(), synchronize_output()
[ ] 移除ref——cnt,或者更改位置！！！！！！ref_cnt bugs!!! 
    ->> model.instances_info()   # has bugs , dont user this
    ->> rich.print(PlayingPhoneModel.get_instances_table())
[ ] 重写配置读取！！！！！def parse_model_config(cfg): 
[ ] numba写cuda核
[ ] numba 统一加速瓶颈
[ ] onnx 模型8bit qat量化
[ ] serve engine solve
[ ] numba speed up 
[ ] model visualization
[ ] 用于多线程的多模型维护容器
[ ] base config 新增，作为基类配置，用于其他配置继承，重新优化配置, 包括component name
[ ] 配置文件中的`module`字段可以修改成为component name!
[ ] 支持跟踪算法 
[x] 支持multi-stream输入
[x] 支持multi-video输入
[x] 支持img和video混合输入
[ ] 支持mps
[ ] 检测结果保存： multi-stream, multi-images, labels


# 23.03.08
[ ] host & device mem class
[ ] cuda-rt
[ ] trt fp16 
[ ] trt int8
[ ] speed test
[ ] fp16 / int8 set_binding_shape directly
 


# 23.02.10
[x] 新增TIMER模块，用于统计时间，支持上下文装饰和函数装饰两种方式
[x] setup_logging模块


# 23.02.13
[x] instance @property
[x] 移除flask中logging模块中的handler
[x] hydra logging 新增
[x] 新增OmegaConf resolver
[x] component name 新增
[x] bug: 构建模型使用verbose参数会导致refcnt多2


# 23.02.14


# 23.02.15
[x] onnx fp16 测试
[x] 测试gpu上，pt和onnx的速度
[x] ort provider分配cuda execuation provider
[x] onnx模型分配device_id,并允许自由切换device_id
[x] numpy + numba + onnx for now!
[x] numpy重写batched_nms（），并且优化加速
[x] 统一模型预测结果的格式
[x] pynvml  -> gpu


# 23.02.16
[x] gpu_info --> checking gpu status
[x] GPU name  -> cuda
[x] 对于公共属性，ensemable Model中也要加入
[x] ensemabled model add device   ----> dataclass, str or list, Union[str, int]
[x] onnx输入数据前处理也要分配device_id,把数据放到deviceid上，----》 IOBinding 


# 23.02.17
[x] 静态人流量检测接口返回人头框
[x] bug: device切换时会出现错误！ fixed! 支持cpu -> cuda, cuda -> cpu
[x] [fixed]bug:   ======> device_id, device_type
    for i in range(10):
        with TIMER(f"{i}-th:"):
            y = model(img)
            # model.to(3)    ==============> may has bug!!
[x] 新增device class
[x] type hint some where
[x] 新增 warmup mudle
[x] 新增INSPECTOR module with `repeat` args


# week tasks
1. 多后端多任务模型推理框架继续编写：讨论功能、更新项目配置解析
2. 老年人残疾人模型：数据标注、模型训练
3. 睡岗模型：数据标注、模型训练
4. 口罩模型部署代码修改、提交并通过测试



# 23.02.20
[x] 新增提交/更新模型方法 allrounder.update() ---> submit()
[x] 老年人 ttt的数据整合
[x] 老年人模型训练，第1.1版本, 训练中
[x] 新增 base_predictor.py
[x] 组会
[x] 多人标签数据整合脚本
[x] 口罩模型封装测试代码



# 23.02.23
[X] 新增批量label修改
[x] img trackbar上添加一个新的trackbar，用来避免错误点击
[x] BUG::: letterbox and stacking!!

