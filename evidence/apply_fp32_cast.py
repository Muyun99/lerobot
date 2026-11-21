"""Toggle the paper's one-line fp32 master-copy fix inside an installed LeRobot.

  python apply_fp32_cast.py on|off|status

It edits `AutoModel.from_config(lm_expert_config)` where SmolVLA builds the action
expert, which is what the paper (arXiv:2610.31859) proposes. `off` restores the file
byte-for-byte, so a checkout can be reused for both arms.
"""
import pathlib
import sys

import lerobot

TARGET = pathlib.Path(lerobot.__file__).parent / "policies" / "smolvla" / "smolvlm_with_expert.py"
ORIG = "        self.lm_expert = AutoModel.from_config(lm_expert_config)\n"
PATCH = (
    "        self.lm_expert = AutoModel.from_config(lm_expert_config).to(torch.float32)"
    "  # fp32 master copy (arXiv:2610.31859)\n"
)


def main(mode):
    s = TARGET.read_text()
    if mode == "status":
        print("patched" if PATCH in s else "clean", TARGET)
        return 0
    if mode == "on":
        if PATCH in s:
            print("already patched")
        elif ORIG in s:
            TARGET.write_text(s.replace(ORIG, PATCH, 1))
            print("patched:", TARGET)
        else:
            print("pattern not found; is this the expected LeRobot version?")
            return 1
    elif mode == "off":
        if ORIG in s:
            print("already clean")
        elif PATCH in s:
            TARGET.write_text(s.replace(PATCH, ORIG, 1))
            print("restored:", TARGET)
        else:
            print("pattern not found")
            return 1
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "status"))
