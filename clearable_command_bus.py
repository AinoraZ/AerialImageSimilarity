from buslane.commands import CommandBus

class ClearableCommandBus(CommandBus):
    def clear(self):
        self._handlers.clear()
