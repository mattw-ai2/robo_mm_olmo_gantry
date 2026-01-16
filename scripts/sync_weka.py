import os
import getpass

# Usage
# python scripts/sync_weka.py


def main():
    dest_adr = f"weka://prior-default/{getpass.getuser()}"

    command = "rclone  sync --progress --copy-links\
         --exclude .idea \
         --exclude __pycache__/ \
         --exclude .DS_Store \
         --exclude .envrc \
         --exclude .venv/ \
         --exclude .git/ \
         --exclude output/ \
         --exclude */static/ \
         --exclude wandb/ \
         --exclude src/ \
         --exclude debug/ \
         --exclude experiment_output/\
         --exclude scratch/ \
         --exclude *.pth\
         ../robo_mm_olmo {}/robo_mm_olmo".format(
        dest_adr
    )
    print(command)
    os.system(command)


if __name__ == "__main__":
    main()