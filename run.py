import argparse
from experiments import train_nudge, train_seaquest

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Run experiments")
    # parser.add_argument('--exp', choices=['kangaroo', 'seaquest'], required=True, help='Which experiment to run')
    # args = parser.parse_args()

    # if args.exp == 'kangaroo':
    train_nudge.main()
    # elif args.exp == 'seaquest':
    # train_seaquest.main()