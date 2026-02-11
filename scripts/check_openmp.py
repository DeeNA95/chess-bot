import sys
import os

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import mcts_cpp
    print(f"Imported mcts_cpp from: {mcts_cpp.__file__}")
    print(f"Directory: {dir(mcts_cpp)}")
    if hasattr(mcts_cpp, 'get_openmp_info'):
        print(mcts_cpp.get_openmp_info())
    else:
        print("get_openmp_info MISSING")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
