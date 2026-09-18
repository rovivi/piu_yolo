"""Diagnóstico de GPU multiplataforma: MPS (Mac Apple Silicon) / ROCm / CUDA / CPU.

Sustituye a simple_check.py. Uso: python check_gpu.py
"""

from gpu import setup_device_env, find_best_weights

setup_device_env()

import platform

import torch


def main():
    print(f"Plataforma: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        backend = 'ROCm/HIP' if torch.version.hip else 'CUDA'
        print(f"✅ GPU disponible ({backend} {torch.version.hip or torch.version.cuda})")
        try:
            n = torch.cuda.device_count()
            print(f"   Dispositivos: {n}")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                print(f"   [{i}] {props.name} — {props.total_memory / 1e9:.1f} GB")
        except Exception as e:
            # torch 2.4.1+rocm6.0 sin amdsmi lanza NameError aquí
            print(f"   (device_count falló: {e} — usar device 0 igualmente)")

    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        print("✅ MPS disponible (Apple Silicon)")

    if not torch.cuda.is_available() and not (mps and mps.is_available()):
        print("⚠️  Sin GPU — se entrenará en CPU (lento)")

    best = find_best_weights()
    print(f"Mejor modelo actual: {best or 'ninguno (entrena primero)'}")


if __name__ == '__main__':
    main()
