"""
Command line for actuneo.ai

Run in a terminal, where the question is asked and the download shows a
progress bar::

    actuneo-ai setup            # recommend a model, ask, then download
    actuneo-ai models           # every model and whether it fits this machine
    actuneo-ai status           # what is installed
    actuneo-ai benchmark        # how well the installed model does
    actuneo-ai remove           # delete the model

``python -m actuneo.ai`` does the same if the ``actuneo-ai`` command is not
on the path.
"""

import argparse
import getpass
import sys
from typing import List, Optional

from .backends import PROVIDERS, BackendError
from .catalog import load_catalog
from .hardware import detect_hardware
from .setup import SetupError, configure, placement, remove_model, select_model, status
from .setup import setup as _setup_function


def _models() -> None:
    """Print every model in the catalog and whether it fits this machine."""
    hardware = detect_hardware()
    catalog = load_catalog()
    selection = select_model(hardware, catalog)
    recommended = selection.recommended.model.key if selection.recommended else None
    print(f"This machine: {hardware.describe()}")
    print()
    print(f"{'Key':20s} {'Model':40s} {'Download':>9s}  Fits this machine")
    for model in catalog:
        comfortable = placement(model, hardware)
        tight = placement(model, hardware, relaxed=True)
        if comfortable is not None:
            fits = f"yes, {comfortable.runs_on()}"
        elif tight is not None:
            fits = "only just, may be slow"
        else:
            fits = "no"
        if model.key == recommended:
            fits += "   <- recommended"
        if model.role != "tier":
            fits += "   (challenger)"
        print(f"{model.key:20s} {model.name:40s} {model.size_gb:6.1f} GB  {fits}")
    print()
    print("Install one with:  actuneo-ai setup --model KEY")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actuneo-ai",
        description="Set up a local language model for ACTUNEO, chosen to suit this machine.")
    commands = parser.add_subparsers(dest="command", metavar="command")

    setup = commands.add_parser(
        "setup", help="recommend a model, ask, then download it (the default)")
    setup.add_argument("--model", help="take this model without asking: a key from "
                       "'actuneo-ai models', or smallest, recommended, largest")
    setup.add_argument("--auto", action="store_true",
                       help="take the recommended model without asking")
    setup.add_argument("--yes", "-y", action="store_true", help="answer yes to the question")
    setup.add_argument("--force", action="store_true",
                       help="choose again even if a model is already installed")

    commands.add_parser("models", help="list the models and whether each fits this machine")
    commands.add_parser("status", help="show what is installed")
    commands.add_parser("benchmark", help="test the installed model on 20 actuarial requests")

    remove = commands.add_parser("remove", help="delete the downloaded model")
    remove.add_argument("--yes", "-y", action="store_true", help="delete without asking")

    use = commands.add_parser("configure", help="use a provider with your own API key instead")
    use.add_argument("provider", choices=PROVIDERS)
    use.add_argument("--model", help="the provider's model to use")
    use.add_argument("--save-key", action="store_true",
                     help="store the key in the system keychain (needs the keyring package)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command line. Returns the exit code."""
    arguments = _parser().parse_args(argv)
    command = arguments.command or "setup"
    try:
        if command == "setup":
            _setup_function(auto=getattr(arguments, "auto", False),
                            model=getattr(arguments, "model", None),
                            yes=getattr(arguments, "yes", False),
                            force=getattr(arguments, "force", False))
        elif command == "models":
            _models()
        elif command == "status":
            status()
        elif command == "benchmark":
            from .benchmark import benchmark
            benchmark()
        elif command == "remove":
            remove_model(yes=arguments.yes)
        elif command == "configure":
            key = None
            if arguments.provider not in ("ollama",):
                # Asked for here so that the key is not left in the shell history
                key = getpass.getpass(
                    f"{arguments.provider} API key (leave empty to use the environment "
                    "variable): ") or None
            configure(arguments.provider, api_key=key, model=arguments.model,
                      save_key=arguments.save_key)
            status()
    except (SetupError, BackendError) as problem:
        print(f"\n{problem}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Run the command again to carry on; a download resumes where "
              "it stopped.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
