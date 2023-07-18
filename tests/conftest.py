def pytest_addoption(parser):
    parser.addoption(
        "--new_task",
        action="store_true",
        help="new_tasks_found",
    )
