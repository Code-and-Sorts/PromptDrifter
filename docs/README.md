# PromptDrifter Documentation

This directory contains the documentation website for PromptDrifter, built with [Docusaurus](https://docusaurus.io/).

## Development

To start the development server:

```bash
npm run start
```

This will start a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

## Build

To build the static files of the website for production:

```bash
npm run build
```

This will generate static content in the `build` directory that can be served by any static hosting service.

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment is handled by the GitHub Actions workflow defined in `.github/workflows/deploy-docs.yml`.

## Project Structure

- `/docs/`: Contains the Markdown files for the documentation
- `/static/`: Contains static assets like images, fonts, etc.
- `/docusaurus.config.ts`: Main configuration file for Docusaurus
- `/sidebars.ts`: Sidebar configuration

## Adding Content

### Documentation

To add new documentation, create Markdown files in the `/docs/` directory. Each file should have a front matter section with at least a `sidebar_position` property to determine its position in the sidebar.

Example:

```md
---
sidebar_position: 2
---

# My New Doc

This is a new documentation page.
```

### Blog Posts

To add new blog posts, create Markdown files in the `/blog/` directory with a front matter section.

Example:

```md
---
slug: my-blog-post
title: My Blog Post
authors: [author_name]
tags: [tag1, tag2]
---

This is my new blog post.
```
