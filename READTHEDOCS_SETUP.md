# Read the Docs Setup Guide for ACTUNEO

This guide will help you set up the documentation on Read the Docs.

## Prerequisites

1. GitHub account with access to the ACTUNEO repository
2. Read the Docs account (free at https://readthedocs.org/)

## Step-by-Step Instructions

### 1. Sign Up / Login to Read the Docs

1. Go to https://readthedocs.org/
2. Click "Sign Up" or "Log In"
3. Connect with your GitHub account

### 2. Import Your Project

1. After logging in, click "Import a Project"
2. Click "Import Manually" or select from your GitHub repositories
3. Fill in the project details:
   - **Name**: `actuneo`
   - **Repository URL**: `https://github.com/ShannonT20/ACTUNEO`
   - **Repository type**: Git
   - **Default branch**: `main` (or your default branch)
   - **Default version**: `latest`
   - **Programming language**: Python

4. Click "Next" or "Finish"

### 3. Configure the Project

1. Go to your project's admin page (https://readthedocs.org/dashboard/actuneo/edit/)
2. Under "Admin" → "Settings":
   - Make sure "Documentation type" is set to "Sphinx"
   - Check "Build documentation with MkDocs" is **unchecked**
   - Set Python interpreter to "CPython 3.11" or higher

3. Under "Admin" → "Advanced Settings":
   - Set "Requirements file" to `docs/requirements.txt` (if not auto-detected)
   - Check "Install your project inside a virtualenv using setup.py install"
   - Save changes

### 4. Build the Documentation

1. Go to "Builds" tab
2. Click "Build version: latest"
3. Wait for the build to complete (usually 2-5 minutes)
4. Check the build log for any errors

### 5. View Your Documentation

Once the build succeeds, your documentation will be available at:
- https://actuneo.readthedocs.io/

### 6. Enable Versions

1. Go to "Versions" tab
2. Make sure "latest" is active
3. You can also activate specific tags/branches for versioned docs

## Troubleshooting

### Build Fails

If the build fails, check:

1. **Missing dependencies**: Ensure `docs/requirements.txt` includes all needed packages
2. **Import errors**: Make sure the package is installed in the build environment
3. **Sphinx errors**: Check for syntax errors in `.rst` files

Common fixes:

```yaml
# .readthedocs.yaml - ensure this configuration is correct
version: 2

sphinx:
  configuration: docs/conf.py

python:
  version: "3.11"
  install:
    - requirements: requirements.txt
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

### Documentation Badge Not Updating

- Clear your browser cache
- Wait a few minutes for CDN to update
- Check badge URL matches your project: `https://readthedocs.org/projects/actuneo/badge/?version=latest`

### Custom Domain (Optional)

To use a custom domain:

1. Go to "Admin" → "Domains"
2. Add your custom domain
3. Follow DNS configuration instructions
4. Wait for SSL certificate to be issued

## Automatic Builds

Read the Docs will automatically rebuild documentation when you:
- Push to your GitHub repository
- Create a new tag/release
- Merge a pull request

You can configure webhooks in "Admin" → "Integrations" if automatic builds aren't working.

## Local Documentation Build

To build documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

Open `docs/_build/html/index.html` in your browser.

## Additional Resources

- Read the Docs Documentation: https://docs.readthedocs.io/
- Sphinx Documentation: https://www.sphinx-doc.org/
- reStructuredText Primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

## Support

If you encounter issues:
- Read the Docs support: https://docs.readthedocs.io/en/stable/support.html
- ACTUNEO issues: https://github.com/ShannonT20/ACTUNEO/issues

