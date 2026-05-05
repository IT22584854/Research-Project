
def detect_category(filename: str) -> str:
    """
    Detect document category from filename.
    """

    filename = filename.upper()

    if "EPID" in filename:
        return "EPID"
    elif "LRH" in filename:
        return "LRH"
    elif "MOH" in filename:
        return "MOH"
    elif "MRI" in filename:
        return "MRI"
    elif "NHK" in filename:
        return "NHK"
    elif "NHSL" in filename:
        return "NHSL"
    else:
        return "GENERAL"