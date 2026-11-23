from sheaf_verify.ledger import Ledger


def latest(store):
    found = {}
    for record in Ledger(store.root).records():
        found[record["gate_id"]] = record
    return found


def criterion(record, identifier):
    for item in record["criteria"]:
        if item["id"] == identifier:
            return item
    raise AssertionError(identifier)
