import socket
import subprocess
import time

LOCALHOST = "localhost"


def get_safe_port():
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def launch_server(default_artifact_root, port=5001):
    cmd = [
        "mlflow",
        "server",
        "--default-artifact-root",
        default_artifact_root,
        "--serve-artifacts",
        "--port",
        str(port),
    ]

    process = subprocess.Popen(cmd)
    await_server_up_or_die(port)
    return process


def await_server_up_or_die(port, timeout=60):
    """Waits until the local flask server is listening on the given port."""
    start_time = time.time()
    connected = False
    while not connected and time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((LOCALHOST, port))
        if result == 0:
            connected = True
        else:
            time.sleep(0.5)
    if not connected:
        raise Exception("Failed to connect on %s:%s after %s seconds" %
                        (LOCALHOST, port, timeout))
