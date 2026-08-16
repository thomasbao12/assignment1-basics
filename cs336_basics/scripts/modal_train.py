import modal
import subprocess

app = modal.App("cs336-a1")

data_volume = modal.Volume.from_name("cs336")

image = (
    modal.Image.debian_slim(python_version="3.12.13")
    .uv_sync()
    .add_local_python_source("cs336_basics")
    .add_local_file(
        "cs336_basics/scripts/configs/tinystories_b200_smoke.toml",
        remote_path="/root/config.toml",
    )
)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/cs336": data_volume},
    timeout=60 * 60,
)
def train():
    subprocess.run(
        [
            "python",
            "-m",
            "cs336_basics.scripts.training_loop",
            "--config",
            "/root/config.toml",
        ],
        check=True,
    )


@app.local_entrypoint()
def main():
    train.remote()