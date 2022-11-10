# C.A.P.E. - Computer Assisted Panel Extractor

Forked from <https://github.com/CodeMinion/C.A.P.E>

**C.A.P.E.** is a Python package used to extract individual panels from comic book pages.

```bash
./cape.py [input file or folder] [output folder]
```

You can add the flag `-d` to replicate the directory structure of the input folder.

### Features

-   Extract individual panels.
-   Save the positions of the panels in a metadata file for use in other programs.

### Summary

C.A.P.E is a smart editor to reduce the time taken to extract comic book panel information from comic books. This extracted information is then stored as part of the comic book digital file and can be used by reader applications to created a guided reading experience for the users. By leveraging the computer extracted information we can hope to reduce the overall time to extract the panels from a comic by a magnitude of what it would take for a human alone.

## Metadata Structure

After the first pass the panel recognizer outputs a metadata file for each of the comic pages analyzed. This metadata is stored alongside each page with the same name and the extension `.cpanel`.

The structure of the metadata file is as follows:

```json
{
  "panels": [{
    "box": {
      "y": 68,
      "x": 28,
      "w": 1073,
      "h": 521
    },
    "shape": [{
      "y": 90,
      "x": 48
    }]
  }, {
    "box": {
      "y": 620,
      "x": 28,
      "w": 521,
      "h": 521
    },
    "shape": []
  }],
  "imagePath": "2013.06.01-Sloan_p01.jpg",
  "version": 2
}
```

-   version: Current version of the format, used for tracking future changes.
-   panels: List of panel information objects. Each object contains all the information for a given panel.
-   box: This is the bounding box coordinates and dimensions of the panel. All the coordinates and dimensions are in source coordinates.
-   shape: List of (x,y) coordinates of each point of the bounding shape of the panel. Used to support more unique panels that don't fit in a box. Not supported as of version 1.
-   imagePath: Relative path to the image of the comic page this metadata is associated with. In metadata since version 2.
