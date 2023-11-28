from train import main as train_main
from test import main as test_main
import sys

def run():
  args = sys.argv
  
  if (len(args) != 3):
    print('Usage: python3 runner.py <test_image_name> <pickle_file_name>')
    return
  
  test_image_name = args[1]
  pkl_file_name = args[2]

  plot = input("Plot? (enter 'y' or 'n'): ").lower()
  while (plot != 'y' and plot != 'n'):
    plot = input("Plot? (enter 'y' or 'n'): ").lower()

  if (plot == 'y'): plot = True
  else: plot = False

  print('Running training file...')
  train_main(plot=plot)

  print('############################################################')
  print('Running test file...')
  test_main(test_image_name, pkl_file_name, plot=plot)

if __name__ == '__main__':
  run()