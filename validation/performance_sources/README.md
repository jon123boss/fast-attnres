# Sealed performance source

`compiled-step-81dffbfeb0f84470513e846e3df8080e8ffb563d.bundle` is a complete Git
bundle of the clean source commit used by the archived six-report compiled-step
campaign. Its SHA-256 is
`09547604f0a9630ed8769cf55479f255754dce2431d325ddcf250af8bafdde17`.

The v1.0.0 release workflow verifies the bundle digest, reconstructs a detached
checkout, verifies its exact HEAD and clean status, and runs the offline report
auditor against that checkout. The bundle keeps the measured commit durable
without adding historical branches or extra commits to the public repository.
It is repository evidence and is excluded from the wheel and source distribution.
