#!/usr/bin/env python3
"""
Mock Confluence Data Generator
Generates sample documentation data for testing the RAG system.
"""

import json
from datetime import datetime

MOCK_CONFLUENCE_PAGES = [
    {
        "id": "12345",
        "title": "Employee Onboarding Guide",
        "version": {"number": 15},
        "body": {
            "storage": {
                "value": """
                <h1>Employee Onboarding Guide</h1>
                <p>Welcome to the company! This guide will help you get started with your first week.</p>

                <h2>First Day Checklist</h2>
                <ul>
                    <li>Complete HR paperwork in the HR portal</li>
                    <li>Set up your company email account</li>
                    <li>Install required software from the IT portal</li>
                    <li>Meet with your manager for orientation</li>
                    <li>Join the #new-hires Slack channel</li>
                </ul>

                <h2>IT Setup</h2>
                <p>All employees receive a laptop and access to the following systems:</p>
                <ul>
                    <li>Email: company-name.com domain</li>
                    <li>VPN: Use Cisco AnyConnect</li>
                    <li>Password Manager: 1Password for Teams</li>
                    <li>Development Tools: GitHub Enterprise access</li>
                </ul>

                <h2>Benefits Overview</h2>
                <p>Full-time employees are eligible for:</p>
                <ul>
                    <li>Health insurance (medical, dental, vision)</li>
                    <li>401(k) with 4% company match</li>
                    <li>20 days PTO per year</li>
                    <li>Professional development budget: $2,000/year</li>
                </ul>

                <p>For questions, contact HR at hr@company.com</p>
                """
            }
        }
    },
    {
        "id": "23456",
        "title": "Engineering Best Practices",
        "version": {"number": 23},
        "body": {
            "storage": {
                "value": """
                <h1>Engineering Best Practices</h1>

                <h2>Code Review Guidelines</h2>
                <p>All code must be reviewed before merging to main branch.</p>
                <ul>
                    <li>Minimum 2 approvals required for production code</li>
                    <li>Use GitHub Pull Requests for all changes</li>
                    <li>PR descriptions must include: purpose, testing done, deployment notes</li>
                    <li>Reviews should be completed within 24 hours</li>
                </ul>

                <h2>Testing Standards</h2>
                <p>We maintain high code quality through comprehensive testing:</p>
                <ul>
                    <li>Unit test coverage must be at least 80%</li>
                    <li>Integration tests required for API endpoints</li>
                    <li>Run tests locally before pushing: <code>npm test</code></li>
                    <li>CI/CD pipeline runs all tests automatically</li>
                </ul>

                <h2>Deployment Process</h2>
                <ol>
                    <li>Merge PR to main branch</li>
                    <li>CI builds and tests automatically</li>
                    <li>Deploy to staging environment for QA</li>
                    <li>After QA approval, deploy to production</li>
                    <li>Monitor error rates and performance metrics</li>
                </ol>

                <h2>Tech Stack</h2>
                <ul>
                    <li>Frontend: React, TypeScript, Tailwind CSS</li>
                    <li>Backend: Node.js, Express, PostgreSQL</li>
                    <li>Infrastructure: Kubernetes, AWS, Docker</li>
                    <li>Monitoring: Datadog, Sentry</li>
                </ul>
                """
            }
        }
    },
    {
        "id": "34567",
        "title": "Remote Work Policy",
        "version": {"number": 8},
        "body": {
            "storage": {
                "value": """
                <h1>Remote Work Policy</h1>

                <h2>Work Arrangements</h2>
                <p>The company supports flexible work arrangements:</p>
                <ul>
                    <li>Fully remote positions available for most roles</li>
                    <li>Hybrid: 2-3 days in office, rest remote</li>
                    <li>Office-based for roles requiring physical presence</li>
                </ul>

                <h2>Communication Expectations</h2>
                <p>When working remotely, team members should:</p>
                <ul>
                    <li>Be available during core hours: 10 AM - 3 PM local time</li>
                    <li>Respond to Slack messages within 2 hours during work hours</li>
                    <li>Keep calendar updated with availability</li>
                    <li>Use video for team meetings when possible</li>
                </ul>

                <h2>Equipment and Stipends</h2>
                <p>Remote employees receive:</p>
                <ul>
                    <li>Company laptop and accessories</li>
                    <li>$500 home office setup stipend</li>
                    <li>$100/month internet reimbursement</li>
                    <li>Co-working space membership if needed</li>
                </ul>

                <h2>Security Requirements</h2>
                <ul>
                    <li>Always use VPN when accessing company systems</li>
                    <li>Enable full-disk encryption on work devices</li>
                    <li>Use password manager for all credentials</li>
                    <li>Never share confidential data over unsecured channels</li>
                </ul>
                """
            }
        }
    },
    {
        "id": "45678",
        "title": "Kubernetes Deployment Guide",
        "version": {"number": 12},
        "body": {
            "storage": {
                "value": """
                <h1>Kubernetes Deployment Guide</h1>

                <h2>Cluster Architecture</h2>
                <p>Our production infrastructure runs on AWS EKS with the following setup:</p>
                <ul>
                    <li>3 availability zones for high availability</li>
                    <li>Node groups: t3.large for general workloads, c5.2xlarge for ML</li>
                    <li>Auto-scaling enabled based on CPU and memory metrics</li>
                    <li>Network policy enforcement with Calico</li>
                </ul>

                <h2>Deployment Process</h2>
                <p>To deploy a new service:</p>
                <ol>
                    <li>Create Helm chart in <code>k8s/charts/&lt;service-name&gt;</code></li>
                    <li>Test locally with minikube: <code>helm install --dry-run</code></li>
                    <li>Deploy to staging: <code>helm upgrade --install &lt;name&gt; ./chart</code></li>
                    <li>Run smoke tests against staging</li>
                    <li>Deploy to production with same command</li>
                </ol>

                <h2>Required Configurations</h2>
                <ul>
                    <li>Resource limits: Always set CPU and memory requests/limits</li>
                    <li>Health checks: Configure liveness and readiness probes</li>
                    <li>Secrets: Store in Kubernetes Secrets, never in code</li>
                    <li>Service mesh: Use Istio for service-to-service communication</li>
                </ul>

                <h2>Monitoring</h2>
                <p>All services must expose:</p>
                <ul>
                    <li>Prometheus metrics at <code>/metrics</code></li>
                    <li>Health endpoint at <code>/health</code></li>
                    <li>Structured JSON logs to stdout</li>
                </ul>
                """
            }
        }
    },
    {
        "id": "56789",
        "title": "Data Privacy and GDPR Compliance",
        "version": {"number": 19},
        "body": {
            "storage": {
                "value": """
                <h1>Data Privacy and GDPR Compliance</h1>

                <h2>Data Classification</h2>
                <p>All data must be classified into one of these categories:</p>
                <ul>
                    <li><strong>Public:</strong> Information that can be freely shared</li>
                    <li><strong>Internal:</strong> Company information for employees only</li>
                    <li><strong>Confidential:</strong> Sensitive business data, limited access</li>
                    <li><strong>Restricted:</strong> PII, financial data, requires encryption</li>
                </ul>

                <h2>GDPR Requirements</h2>
                <p>When handling EU customer data:</p>
                <ul>
                    <li>Obtain explicit consent before collection</li>
                    <li>Provide clear privacy notice and data usage policy</li>
                    <li>Enable data export (right to data portability)</li>
                    <li>Support deletion requests within 30 days</li>
                    <li>Report breaches within 72 hours</li>
                </ul>

                <h2>Data Retention</h2>
                <ul>
                    <li>Customer data: Retained while account is active + 90 days</li>
                    <li>Logs and analytics: 13 months maximum</li>
                    <li>Financial records: 7 years (legal requirement)</li>
                    <li>Employee records: Duration of employment + 3 years</li>
                </ul>

                <h2>Security Controls</h2>
                <ul>
                    <li>Encryption at rest: AES-256 for all databases</li>
                    <li>Encryption in transit: TLS 1.3 for all connections</li>
                    <li>Access control: Role-based with principle of least privilege</li>
                    <li>Audit logging: All data access must be logged</li>
                </ul>

                <p>Questions? Contact the Data Protection Officer at dpo@company.com</p>
                """
            }
        }
    }
]


def get_mock_pages():
    """Return the list of mock page data."""
    return MOCK_CONFLUENCE_PAGES


if __name__ == "__main__":
    print(f"Loaded {len(MOCK_CONFLUENCE_PAGES)} mock pages.")