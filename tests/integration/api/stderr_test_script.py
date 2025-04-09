#!/usr/bin/env python
"""
Simple script that outputs messages to stderr for testing.
"""
import sys

# Write complete lines
sys.stderr.write("Error line 1\n")
sys.stderr.flush()

# Write partial line then complete it
sys.stderr.write("Error line 2 part 1")
sys.stderr.flush()
sys.stderr.write(" part 2\n")
sys.stderr.flush()

# Another complete line
sys.stderr.write("Final error line\n")
sys.stderr.flush()