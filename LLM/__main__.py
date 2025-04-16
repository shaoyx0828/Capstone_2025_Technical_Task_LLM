import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from model.train_qwen_sentiment import main

if __name__ == "__main__":
    main()
