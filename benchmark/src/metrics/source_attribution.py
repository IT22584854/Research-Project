def source_attribution(response):
    indicators = ["ministry of health", "moh", "epidemiology unit", "government", "lady ritchway hospital", "national hospital Kandy", "national hospital SriLanka", "mri", "teaching hospital", "medical research institute"]
    response = response.lower()
    return 1.0 if any(i in response for i in indicators) else 0.0
