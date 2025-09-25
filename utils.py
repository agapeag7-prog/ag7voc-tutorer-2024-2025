def call_intent_command(intent, command_text=None):
    from ag7voc import (
        read_file, write_file, delete_file, create_folder, list_files,
        rename_file, move_file, show_events, add_event, delete_event, modify_event,
        get_system_info, launch_app, shutdown_computer, restart_computer, lock_computer,
        search_web, send_email, show_help, process_voice_command
    )

    intent_map = {
        "read_file": read_file,
        "write_file": write_file,
        "delete_file": delete_file,
        "create_folder": create_folder,
        "list_files": list_files,
        "rename_file": rename_file,
        "move_file": move_file,
        "show_events": show_events,
        "add_event": add_event,
        "delete_event": delete_event,
        "modify_event": modify_event,
        "get_time": lambda: process_voice_command("il est quelle heure"),
        "get_date": lambda: process_voice_command("quelle date"),
        "show_help": show_help,
        "launch_app": lambda: launch_app(command_text or "explorateur"),
        "shutdown": shutdown_computer,
        "restart": restart_computer,
        "lock": lock_computer,
        "search_web": lambda: search_web(command_text or ""),
        "send_email": lambda: send_email(command_text or ""),
        "system_info": get_system_info,
    }

    func = intent_map.get(intent)
    if func:
        try:
            if intent in ["write_file", "add_event", "delete_event", "modify_event", "search_web", "send_email", "launch_app"]:
                return func(command_text)
            else:
                return func()
        except Exception as e:
            return f"Erreur: {e}"
    return f"Intention '{intent}' non reconnue"