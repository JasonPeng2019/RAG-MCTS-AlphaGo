# Extra Files Directory

This directory contains supplementary files needed for the DataGo RAG-MCTS system but should be deployed outside the main RAG-MCTS-AlphaGo folder.

## Contents

### 1. `configs/` - KataGo Configuration Files

**Purpose:** Custom KataGo configuration files that specify visit counts and other engine parameters.

**Deployment:**
- Copy this folder to your KataGo installation directory
- Path should be: `/path/to/KataGo/configs/`
- Reference these configs when starting KataGo via GTP

**Files:**
- `gtp_800visits.cfg` - KataGo config with 800 visits for base gameplay
  - Used as the starting configuration
  - DataGo dynamically adjusts visits to 2000 for deep searches via `kata-set-param maxVisits`

**Usage Example:**
```bash
/path/to/KataGo/cpp/katago gtp \
    -model /path/to/model.bin.gz \
    -config /path/to/KataGo/configs/gtp_800visits.cfg
```

### 2. `Go_env/` - Python Virtual Environment

**Purpose:** Isolated Python environment with all dependencies for DataGo.

**Recommended Structure (current workspace):**
```
RAGFlow-Datago/
├── Go_env/                      # ← Virtualenv lives alongside the codebase (ignored by git)
├── katago_repo/                 # Bundled KataGo build + configs
├── src/, testing/, rag_store/, …
└── extra_files/                 # Archive of Go_env + configs for redeployment
```

**Why keep it separate?**
- Virtual environments are large (~500MB+)
- Should not be version controlled
- Can be shared across multiple projects
- Easier to recreate/update independently

**Activation (assuming the layout above):**
```bash
source Go_env/bin/activate
```

**Dependencies Included:**
- Python 3.12.3
- numpy 2.3.4
- pyyaml 6.0.3
- Other packages listed in `requirements.txt`

## Setup Instructions

### First-Time Setup

1. **Deploy KataGo Configs:**
   ```bash
   cp -r extra_files/configs /path/to/KataGo/
   ```

2. **Move Virtual Environment (if you use the archived copy):**
   ```bash
   mv extra_files/Go_env Go_env
   ```
   Or create a fresh one next to the repo:
   ```bash
   cd ..
   python3 -m venv Go_env
   source Go_env/bin/activate
   pip install -r RAGFlow-Datago/requirements.txt
   ```

3. **Update Config Paths:**
   Edit `src/bot/config.yaml` to point to correct paths:
   ```yaml
   katago:
     executable_path: "./katago_repo/KataGo/cpp/build-opencl/katago"
     model_path: "./katago_repo/run/default_model.bin.gz"
     config_path: "./katago_repo/run/gtp_800visits.cfg"
   ```

### Verification

Test that everything is configured correctly:

```bash
# Activate environment
source Go_env/bin/activate

# Run quick test
cd RAGFlow-Datago
./testing/quick_competitive_test.sh
```

Expected output:
- DataGo should beat KataGo with ~90% win rate
- Should see "Set maxVisits to 800" and "Set maxVisits to 2000" in logs

## Notes

- **Do not commit Go_env/** to git (already in `.gitignore`)
- The `configs/` folder can be version controlled as it's small and project-specific
- If KataGo configs change, update `extra_files/configs/` and redeploy
- Virtual environment may need recreation when Python version changes

## Troubleshooting

**Problem:** KataGo not found
```bash
# Check KataGo path in config.yaml
# Verify executable exists:
ls -la /path/to/KataGo/cpp/katago
```

**Problem:** Config file not found
```bash
# Verify configs are in KataGo directory:
ls -la /path/to/KataGo/configs/gtp_800visits.cfg
```

**Problem:** Import errors
```bash
# Recreate virtual environment:
cd ..
rm -rf Go_env
python3 -m venv Go_env
source Go_env/bin/activate
cd RAGFlow-Datago
pip install -r requirements.txt
```

**Problem:** Visit count not changing
```bash
# Check that kata-set-param is working:
python3 -c "
from src.bot.gtp_controller import GTPController
import logging
logging.basicConfig(level=logging.DEBUG)
katago = GTPController([...])  # Your KataGo command
katago.set_max_visits(2000)
katago.quit()
"
```

## Related Documentation

- **VISIT_COUNTS_EXPLAINED.md** - How DataGo uses 800 vs 2000 visits
- **COMPETITIVE_RESULTS.md** - Performance results showing 90% win rate
- **RECURSIVE_DEEP_SEARCH.md** - Architecture of adaptive search system
- **src/bot/config.yaml** - Main configuration file with all parameters
