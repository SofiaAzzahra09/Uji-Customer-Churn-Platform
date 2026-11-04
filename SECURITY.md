# 🛡️ Security Policy — Customer Churn Prediction System

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :x:                |

---

## 🔒 Security Architecture Highlights

1. **Session & Cookie Hardening:**
   - Session cookies configured with `HttpOnly` and `SameSite=Lax` to prevent session hijacking and cross-site request forgery.
2. **OWASP Compliance:**
   - Security headers enforced on every HTTP response:
     - `X-Content-Type-Options: nosniff`
     - `X-Frame-Options: SAMEORIGIN`
     - `X-XSS-Protection: 1; mode=block`
     - `Referrer-Policy: strict-origin-when-cross-origin`
3. **Audit Trail:**
   - Every request is tagged with a UUID4 `X-Request-ID` for end-to-end transaction logging.
4. **Least-Privilege Docker Execution:**
   - Production container runs as unprivileged user `appuser:1000`.
5. **Secret Isolation:**
   - Secrets and session keys are loaded exclusively from `.env` and omitted from version control.

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability:

1. **Do NOT open a public GitHub issue.**
2. Email your report directly to: **sofiaazzahra@alumni.ac.id**
3. Please include reproduction steps and expected impact. A response will be provided within 48 hours.
