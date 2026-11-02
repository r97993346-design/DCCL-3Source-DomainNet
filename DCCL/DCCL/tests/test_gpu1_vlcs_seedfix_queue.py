"""Run the shared PID-only queue integration checks for VLCS on GPU1."""
import unittest

import test_gpu0_seedfix_queue as shared


class GPU1VLCSSeedFixQueueTests(shared.GPU0SeedFixQueueTests):
    script = shared.ROOT / 'scripts/run_gpu1_vlcs_seedfix_6exp_after_pid.sh'
    dataset = 'VLCS'
    gpu = '1'
    wait_pid = 3114122


if __name__ == '__main__':
    unittest.main()
