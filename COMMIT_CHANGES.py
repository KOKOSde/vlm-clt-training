#!/usr/bin/env python3
"""Commit and push changes to GitHub."""

import subprocess
import os

os.chdir('/scratch/fkalghan/vlm-clt-training')

print("Committing changes...")
subprocess.call(['git', 'add', '-A'])
result = subprocess.call([
    'git', 'commit', '-m', 
    'Surgical VLM support: minimal updates for CLT training\n\n'
    '- Added VLM detection (AutoModelForVision2Seq)\n'
    '- Pass pixel_values in all forward calls\n'
    '- Support VLM model structure\n'
    '- Only 2 files modified, 6 changes total'
])

if result == 0:
    print("✅ Committed successfully!")
    print("Pushing to GitHub...")
    subprocess.call(['git', 'push', 'origin', 'main'])
    print("✅ Pushed to GitHub!")
else:
    print("⚠️  Nothing to commit or commit failed")

print("\nLatest commit:")
subprocess.call(['git', 'log', '--oneline', '-1'])

