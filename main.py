from utils.allocateGPU import *
try:
    allocate_gpu()
except :
    print("Running Without GPU")
import parser
import _main

if __name__ == "__main__":
    args = parser.parse_args()
    print("#" * 64)
    for i in vars(args):
        print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    print("#" * 64)
    _main.main(args)
