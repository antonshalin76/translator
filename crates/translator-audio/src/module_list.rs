use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct PactlModule {
    pub name: String,
    pub argument: String,
}

#[derive(Debug)]
pub(crate) struct ModuleListError;

pub(crate) fn parse_module_list(
    bytes: &[u8],
) -> Result<HashMap<u32, PactlModule>, ModuleListError> {
    let text = std::str::from_utf8(bytes).map_err(|_| ModuleListError)?;
    let mut modules = HashMap::new();
    if text.is_empty() {
        return Ok(modules);
    }
    if !text.ends_with("\t\n") {
        return Err(ModuleListError);
    }
    for record in text.split_terminator("\t\n") {
        let mut fields = record.splitn(3, '\t');
        let raw_id = fields.next().ok_or(ModuleListError)?;
        let id = raw_id.parse::<u32>().map_err(|_| ModuleListError)?;
        if id == u32::MAX || id.to_string() != raw_id {
            return Err(ModuleListError);
        }
        let name = fields
            .next()
            .filter(|name| !name.is_empty() && !name.contains(['\n', '\r']))
            .ok_or(ModuleListError)?;
        let argument = fields.next().ok_or(ModuleListError)?;
        let module = PactlModule {
            name: name.to_owned(),
            argument: argument.to_owned(),
        };
        if modules.insert(id, module).is_some() {
            return Err(ModuleListError);
        }
    }
    Ok(modules)
}

#[cfg(test)]
mod tests {
    use super::parse_module_list;

    #[test]
    fn renderer_records_preserve_foreign_multiline_and_empty_arguments() {
        let modules = parse_module_list(b"70\tmodule-null-sink\tsink_name=owned\t\n2\tmodule-foreign\t{\nkey=x\ty\n}\t\n0\tmodule-empty\t\t\n4294967294\tmodule-high\targ\t\n").unwrap();
        assert_eq!(modules.len(), 4);
        assert_eq!(modules[&70].name, "module-null-sink");
        assert_eq!(modules[&70].argument, "sink_name=owned");
        assert_eq!(modules[&2].argument, "{\nkey=x\ty\n}");
        assert_eq!(modules[&0].argument, "");
        assert_eq!(modules[&4_294_967_294].name, "module-high");
        assert!(parse_module_list(b"").unwrap().is_empty());
    }

    #[test]
    fn malformed_nonempty_inventory_never_becomes_absence() {
        for payload in [
            &b"\xff"[..],
            &b"[]"[..],
            &b" \n"[..],
            &b"7\tmodule\targs\n"[..],
            &b"7\tmodule\targs\t"[..],
            &b"7\tmodule\targs\t\ntruncated"[..],
            &b"7\tmodule\targs\t\n7\tmodule-other\targs\t\n"[..],
            &b"7\t\targs\t\n"[..],
            &b"7\tmodule\nother\targs\t\n"[..],
            &b"7\tmodule\rother\targs\t\n"[..],
            &b"7\tmodule\t\n"[..],
            &b"7\t\n"[..],
        ] {
            assert!(
                parse_module_list(payload).is_err(),
                "accepted malformed framing"
            );
        }
        for id in [
            "+7",
            "-1",
            "07",
            " 7",
            "7 ",
            "\u{0667}",
            "4294967295",
            "4294967296",
            "",
        ] {
            let payload = format!("{id}\tmodule\targs\t\n");
            assert!(
                parse_module_list(payload.as_bytes()).is_err(),
                "accepted noncanonical ID"
            );
        }
    }
}
