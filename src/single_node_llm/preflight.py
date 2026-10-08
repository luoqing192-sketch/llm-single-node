"""环境自检：打印硬件、PyTorch 和推荐配置。"""  # 模块说明

from __future__ import annotations  # 允许延后解析类型注解

import json  # 把自检结果格式化成 JSON
import os  # 读取 CPU 核数和 POSIX 内存信息
import platform  # 读取操作系统和 Python 版本

import torch  # 检查 PyTorch 与 CUDA


def main() -> None:  # 入口：汇总环境信息
    """收集本机信息，帮助选择 cpu-smoke / cpu-local / cuda 配置。"""  # 函数说明
    memory = None  # 默认拿不到内存大小
    try:  # POSIX 接口在 Windows 上会失败
        pages = os.sysconf("SC_PHYS_PAGES")  # 物理页数量
        page_size = os.sysconf("SC_PAGE_SIZE")  # 每页字节数
        memory = round(pages * page_size / 1024**3, 1)  # 换算成 GiB
    except (AttributeError, ValueError):  # Windows 或接口不可用
        pass  # 保持 memory=None
    result = {  # 组装给用户看的字典
        "platform": platform.platform(),  # 操作系统字符串
        "python": platform.python_version(),  # Python 版本
        "cpu_count": os.cpu_count(),  # 逻辑 CPU 数
        "memory_gib": memory,  # 内存 GiB，可能为 None
        "torch": torch.__version__,  # PyTorch 版本
        "cuda_available": torch.cuda.is_available(),  # 是否有 CUDA
        "cuda_version": torch.version.cuda,  # CUDA 运行时版本
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,  # GPU 名称
        "recommendation": (  # 推荐用哪份配置
            "cuda-1.5b.yaml"  # 有 GPU 走 1.5B 配置
            if torch.cuda.is_available()  # 判断是否有 CUDA
            else "cpu-smoke.yaml first, then cpu-local.yaml"  # 无 GPU 先冒烟再 0.5B
        ),  # recommendation 结束
    }  # result 结束
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 打印 JSON


if __name__ == "__main__":  # 直接运行本文件时
    main()  # 执行自检
