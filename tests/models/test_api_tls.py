"""Exercise certificate verification against a local HTTPS completions server."""

import json
import shutil
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
import requests
from urllib3.exceptions import InsecureRequestWarning

from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


@pytest.fixture(scope="module")
def https_completions_server(tmp_path_factory):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required to generate a temporary test certificate")
    directory = tmp_path_factory.mktemp("api-tls")
    certificate = directory / "certificate.pem"
    key = directory / "key.pem"
    subprocess.run(  # noqa: S603 - resolved openssl executable, fixed arguments
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,digitalSignature,keyEncipherment",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    ca_directory = directory / "ca-directory"
    ca_directory.mkdir()
    shutil.copyfile(certificate, ca_directory / "certificate.pem")
    subprocess.run(  # noqa: S603 - resolved openssl executable, fixed arguments
        [openssl, "rehash", str(ca_directory)],
        check=True,
        capture_output=True,
        timeout=10,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "answer",
                            "message": {"role": "assistant", "content": "answer"},
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield (
                f"https://localhost:{server.server_port}/v1/completions",
                str(certificate),
                str(ca_directory),
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("num_concurrent", [1, 2], ids=["sync", "async"])
@pytest.mark.parametrize("api_class", [LocalCompletionsAPI, LocalChatCompletion])
@pytest.mark.parametrize(
    "verification",
    [
        "custom-ca",
        "custom-ca-dir",
        "default-ca",
        "wrong-host",
        "wrong-host-dir",
        "disabled",
    ],
)
def test_completions_certificate_verification(
    https_completions_server, monkeypatch, num_concurrent, api_class, verification
):
    base_url, certificate, ca_directory = https_completions_server
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    ca_cert_path = None
    if verification in {"custom-ca", "wrong-host"}:
        ca_cert_path = certificate
    elif verification in {"custom-ca-dir", "wrong-host-dir"}:
        ca_cert_path = ca_directory
    elif verification == "disabled":
        ca_cert_path = "unused-missing-ca.pem"
    if verification in {"wrong-host", "wrong-host-dir"}:
        base_url = base_url.replace("localhost", "127.0.0.1")
    api = api_class(
        base_url=base_url,
        model="test-model",
        tokenizer_backend=None,
        num_concurrent=num_concurrent,
        max_retries=1,
        timeout=5,
        verify_certificate=verification != "disabled",
        ca_cert_path=ca_cert_path,
    )
    prompt = (
        [{"role": "user", "content": "Hello"}]
        if api_class is LocalChatCompletion
        else "Hello"
    )
    request = Instance(
        request_type="generate_until", doc={}, arguments=(prompt, {}), idx=0
    )

    if verification in {"default-ca", "wrong-host", "wrong-host-dir"}:
        with pytest.raises(
            (requests.exceptions.SSLError, aiohttp.ClientConnectorCertificateError)
        ):
            api.generate_until([request])
    elif verification == "disabled" and num_concurrent == 1:
        with pytest.warns(InsecureRequestWarning):
            assert api.generate_until([request]) == ["answer"]
    else:
        assert api.generate_until([request]) == ["answer"]
