# tests/test_cli.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli import format_dashboard

def test_format_dashboard():
    stats = {
        'scheduler': {
            'pending': 2,
            'active': 3,
            'completed': 10,
            'pages_used': 24,
            'pages_free': 232,
        },
        'running': True,
    }
    output = format_dashboard(stats, uptime=30.5, throughput=12.3)
    assert 'vLLM' in output or 'Inference' in output
    assert '2' in output  # pending count
    assert '3' in output  # active count
