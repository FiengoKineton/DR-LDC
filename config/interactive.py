def ask_yes_no(prompt: str, default: bool | None = None) -> bool:
    while True:
        if default is None:
            suffix = " [y/n]: "
        else:
            suffix = " [Y/n]: " if default else " [y/N]: "

        ans = input(prompt + suffix).strip().lower()

        if not ans and default is not None:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False

        print("Please answer with y or n.")


def ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    choices_str = ", ".join(choices)
    while True:
        shown = f"{prompt} ({choices_str})"
        if default is not None:
            shown += f" [default: {default}]"
        shown += ": "

        ans = input(shown).strip()
        if not ans and default is not None:
            return default
        if ans in choices:
            return ans

        print(f"Invalid choice. Choose one of: {choices_str}")


def ask_string(prompt: str, default: str | None = None) -> str:
    shown = prompt
    if default is not None:
        shown += f" [default: {default}]"
    shown += ": "

    ans = input(shown).strip()
    return ans if ans else (default if default is not None else "")


def set_method_params(p: dict, method_name: str, young_variant: str | None = None) -> None:
    """
    Overwrite the method-related parameters according to the selected solver family.
    """
    if method_name == "Baseline":
        p["upd"] = 0
        p["FROM_DATA"] = 0
        # leave the others unchanged or set safe defaults
        p["non_convex"] = 0
        p["estm_only"] = 0
        p["old_upd"] = 1
        p["approach"] = "Baseline"

    elif method_name == "DeePC":
        p["upd"] = 1
        p["FROM_DATA"] = 1
        p["non_convex"] = 0
        p["estm_only"] = 0
        p["old_upd"] = 1
        p["approach"] = "DeePC"

    elif method_name == "Young":
        p["upd"] = 1
        p["FROM_DATA"] = 1
        p["non_convex"] = 0
        p["estm_only"] = 0
        p["old_upd"] = 1
        if young_variant == "Iso":
            p["approach"] = "Young"
        elif young_variant == "Dir":
            p["approach"] = "Mats"
        else:
            raise ValueError("Young method requires young_variant to be 'Iso' or 'Dir'.")

    elif method_name == "Young_Schur":
        p["upd"] = 1
        p["FROM_DATA"] = 1
        p["non_convex"] = 0
        p["estm_only"] = 0
        p["old_upd"] = 0
        # approach is not central here, but keep a valid non-DeePC value
        p["approach"] = "Young_Schur"

    elif method_name == "Estm":
        p["upd"] = 1
        p["FROM_DATA"] = 1
        p["non_convex"] = 0
        p["estm_only"] = 1
        # not essential, but keep consistent defaults
        p["old_upd"] = 1
        p["approach"] = "Estm"

    elif method_name == "WFL":
        p["upd"] = 1
        p["FROM_DATA"] = 1
        p["non_convex"] = 1
        p["estm_only"] = 0
        p["old_upd"] = 1
        p["approach"] = "WFL"

    else:
        raise ValueError(f"Unknown method: {method_name}")


def apply_terminal_overrides(cfg: dict) -> dict:
    """
    Interactively overwrite selected config entries after loading the YAML config.
    """
    if "params" not in cfg or not isinstance(cfg["params"], dict):
        raise ValueError("cfg must contain a top-level 'params' dictionary")

    p = cfg["params"]

    # Ensure nested dictionaries exist
    if "directories" not in p or not isinstance(p["directories"], dict):
        p["directories"] = {}

    print("\n=== Interactive configuration ===")

    full_run = ask_yes_no("Do you want to run the full thing?", default=bool(p.get("ALL", 1)))
    p["ALL"] = int(full_run)

    # Reset mutually exclusive analysis flags by default
    p["COST"] = 0
    p["SNR"] = 0
    p["FIND"] = 0

    single_run = False
    analysis_mode = False

    if not full_run:
        mode = ask_choice(
            "Select mode",
            ["single", "analysis"],
            default="single"
        )

        if mode == "single":
            single_run = True
            from_data = ask_yes_no(
                "Run FROM_DATA?",
                default=bool(p.get("FROM_DATA", 1))
            )
            p["FROM_DATA"] = int(from_data)

        elif mode == "analysis":
            analysis_mode = True
            analysis_type = ask_choice(
                "Select analysis",
                ["COST", "SNR", "FIND"],
                default="COST"
            )

            p["COST"] = int(analysis_type == "COST")
            p["SNR"]  = int(analysis_type == "SNR")
            p["FIND"] = int(analysis_type == "FIND")

    # ask correlated / independent when relevant
    if full_run or single_run:
        corr_mode = ask_choice(
            "Use correlated or independent?",
            ["correlated", "independent"],
            default=str(p.get("model", "independent"))
        )
        p["model"] = corr_mode
        
    # Ask runID if ALL or single run
    if full_run or single_run:
        current_runid = p["directories"].get("runID", "test_run")
        p["directories"]["runID"] = ask_string("Which runID?", default=current_runid)

    # Ask method if:
    # - ALL
    # - single run with FROM_DATA
    if full_run or (single_run and int(p.get("FROM_DATA", 0)) == 1):
        method = ask_choice(
            "Which method?",
            ["Baseline", "DeePC", "Young", "Young_Schur", "Estm", "WFL"],
            default="Young"
        )
        if method == "Young":
            young_variant = ask_choice(
                "Young method: Iso or Dir?",
                ["Iso", "Dir"],
                default="Iso"
            )
            set_method_params(p, method, young_variant=young_variant)
        else:
            set_method_params(p, method)


    print("\n=== Final overridden params ===")
    print(f"ALL={p.get('ALL')}, FROM_DATA={p.get('FROM_DATA')}, "
          f"COST={p.get('COST')}, SNR={p.get('SNR')}, FIND={p.get('FIND')}")
    print(f"runID={p.get('directories', {}).get('runID')}")
    print(f"upd={p.get('upd')}, non_convex={p.get('non_convex')}, "
          f"estm_only={p.get('estm_only')}, old_upd={p.get('old_upd')}, "
          f"approach={p.get('approach')}")

    return cfg


def validate_cfg(cfg: dict) -> None:
    p = cfg["params"]

    analysis_flags = [int(p.get("COST", 0)), int(p.get("SNR", 0)), int(p.get("FIND", 0))]
    if sum(analysis_flags) > 1:
        raise ValueError("Only one among COST, SNR, FIND can be 1.")

    if int(p.get("ALL", 0)) == 1 and sum(analysis_flags) > 0:
        raise ValueError("ALL=1 should not be combined with analysis flags.")

    if "directories" not in p or "runID" not in p["directories"]:
        raise ValueError("Missing params['directories']['runID'].")
    