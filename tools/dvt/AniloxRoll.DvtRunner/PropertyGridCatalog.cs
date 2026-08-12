using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Web.Script.Serialization;

namespace AniloxRoll.DvtRunner
{
    internal sealed class PropertyGridCatalog
    {
        public List<PropertyGridCatalogEntry> Properties { get; set; } =
            new List<PropertyGridCatalogEntry>();

        public static PropertyGridCatalog Load()
        {
            string path = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "PropertyGridCoverage.json");
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    "PropertyGrid DVT catalog was not found.", path);

            var serializer = new JavaScriptSerializer();
            PropertyGridCatalog catalog = serializer.Deserialize<PropertyGridCatalog>(
                File.ReadAllText(path, Encoding.UTF8));
            if (catalog == null || catalog.Properties == null ||
                catalog.Properties.Count == 0)
                throw new InvalidDataException(
                    "PropertyGrid DVT catalog is empty: " + path);
            catalog.ValidateCatalog();
            return catalog;
        }

        public IReadOnlyList<PropertyGridCatalogEntry> GetGroup(string group)
        {
            List<PropertyGridCatalogEntry> matches = Properties
                .Where(item => string.Equals(
                    item.Group, group, StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (matches.Count == 0)
                throw new InvalidDataException(
                    "PropertyGrid DVT group was not found: " + group);
            return matches;
        }

        public string AuditProductAssembly(string appExePath)
        {
            if (!File.Exists(appExePath))
                throw new FileNotFoundException(
                    "Product executable was not found.", appExePath);

            string directory = Path.GetDirectoryName(appExePath);
            ResolveEventHandler resolver = (sender, args) =>
            {
                string dependency = Path.Combine(
                    directory,
                    new AssemblyName(args.Name).Name + ".dll");
                return File.Exists(dependency)
                    ? Assembly.LoadFrom(dependency)
                    : null;
            };

            AppDomain.CurrentDomain.AssemblyResolve += resolver;
            try
            {
                Assembly assembly = Assembly.LoadFrom(appExePath);
                Type settingsType = assembly.GetType(
                    "AniloxRoll.Monitor.Core.Data.InspectionSettings",
                    true);
                var editable = TypeDescriptor.GetProperties(settingsType)
                    .Cast<PropertyDescriptor>()
                    .Where(property => property.IsBrowsable && !property.IsReadOnly)
                    .ToList();
                var catalogByName = Properties.ToDictionary(
                    item => item.Name, StringComparer.Ordinal);

                string[] missing = editable
                    .Where(property => !catalogByName.ContainsKey(property.Name))
                    .Select(property => property.Name)
                    .ToArray();
                string[] retired = Properties
                    .Where(item => editable.All(property =>
                        !string.Equals(
                            property.Name, item.Name, StringComparison.Ordinal)))
                    .Select(item => item.Name)
                    .ToArray();
                if (missing.Length > 0 || retired.Length > 0)
                {
                    throw new InvalidDataException(
                        "PropertyGrid DVT catalog mismatch. missing=[" +
                        string.Join(",", missing) + "] retired=[" +
                        string.Join(",", retired) + "]");
                }

                var seenDisplayNames = new Dictionary<string, int>(
                    StringComparer.Ordinal);
                foreach (PropertyDescriptor property in editable)
                {
                    PropertyGridCatalogEntry item = catalogByName[property.Name];
                    int occurrence;
                    seenDisplayNames.TryGetValue(property.DisplayName, out occurrence);
                    seenDisplayNames[property.DisplayName] = occurrence + 1;
                    if (!string.Equals(
                            property.DisplayName,
                            item.DisplayName,
                            StringComparison.Ordinal) ||
                        !string.Equals(
                            property.Category,
                            item.Category,
                            StringComparison.Ordinal) ||
                        item.Occurrence != occurrence)
                    {
                        throw new InvalidDataException(
                            "PropertyGrid DVT metadata mismatch for " +
                            property.Name + ": product=" + property.Category +
                            "/" + property.DisplayName + "#" + occurrence +
                            " catalog=" + item.Category + "/" +
                            item.DisplayName + "#" + item.Occurrence);
                    }
                }

                return "editable=" + editable.Count +
                    " catalog=" + Properties.Count + " missing=0 retired=0";
            }
            finally
            {
                AppDomain.CurrentDomain.AssemblyResolve -= resolver;
            }
        }

        private void ValidateCatalog()
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (PropertyGridCatalogEntry item in Properties)
            {
                if (item == null ||
                    string.IsNullOrWhiteSpace(item.Name) ||
                    string.IsNullOrWhiteSpace(item.DisplayName) ||
                    string.IsNullOrWhiteSpace(item.Category) ||
                    string.IsNullOrWhiteSpace(item.Group) ||
                    string.IsNullOrWhiteSpace(item.Contract))
                    throw new InvalidDataException(
                        "Every PropertyGrid DVT item requires name, display, " +
                        "category, group, and contract.");
                if (!names.Add(item.Name))
                    throw new InvalidDataException(
                        "Duplicate PropertyGrid DVT item: " + item.Name);
                if (item.Occurrence < 0)
                    throw new InvalidDataException(
                        "Property occurrence cannot be negative: " + item.Name);
                if (item.TestValues == null ||
                    item.TestValues.Where(value => value != null).Distinct().Count() < 2)
                    throw new InvalidDataException(
                        "PropertyGrid DVT item requires two distinct test values: " +
                        item.Name);
            }
        }
    }

    internal sealed class PropertyGridCatalogEntry
    {
        public string Name { get; set; }
        public string DisplayName { get; set; }
        public string Category { get; set; }
        public int Occurrence { get; set; }
        public string Group { get; set; }
        public string Contract { get; set; }
        public List<string> TestValues { get; set; } = new List<string>();
    }
}
