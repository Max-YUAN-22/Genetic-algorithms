## FAQ

### Q: pre-commit 报错不是 Git 仓库？
A: 先 `git init && git add . && git commit -m 'init'`，再 `pre-commit install`。

### Q: Apple Silicon 上 torch/torchvision 兼容问题？
A: 参考 README 的 CPU/MPS 轮子安装指引，确保版本匹配。

### Q: 跑不动/显存不足？
A: 使用较小的输入分辨率、批大小；或改为 CPU 演示模式以验证流程。

### Q: 复现实验如何导出？
A: 使用 MLflow：`--mlflow --mlflow_uri file:./mlruns`，然后 `mlflow ui` 查看与导出。


